// Copyright 2019 Guillaume AUJAY. All rights reserved.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "commandline_handler.h"

#include "plot_parameters.h"
#include "benchmark_results.h"
#include "result_parser.h"

#include "plotter_linechart.h"
#include "plotter_barchart.h"
#include "plotter_boxchart.h"
#include "plotter_3dbars.h"
#include "plotter_3dsurface.h"

#include <QApplication>
#include <QFileInfo>
#include <QDebug>

const char* ct_name = "chart-type";
const char* cx_name = "chart-x";
const char* cy_name = "chart-y";
const char* cz_name = "chart-z";
const char* fa_name = "append";
const char* fo_name = "overwrite";


CommandLineHandler::CommandLineHandler()
{
    // Parser configuration
    mParser.setApplicationDescription("JOMT - Help");
    mParser.addHelpOption();
    mParser.addVersionOption();
    mParser.addPositionalArgument("file", "Benchmark results file in json to parse.", "[file]");
    
    QCommandLineOption chartTypeOption(QStringList() << "ct" << ct_name,
               "Chart type (e.g. Lines, Boxes, 3DBars)", "chart_type", "Lines");
    mParser.addOption(chartTypeOption);
    
    QCommandLineOption chartXOption(QStringList() << "cx" << cx_name,
               "Chart X-axis (e.g. a1, t2)", "chart_x", "a1");
    mParser.addOption(chartXOption);
    
    QCommandLineOption chartYOption(QStringList() << "cy" << cy_name,
               "Chart Y-axis (e.g. CPUTime, Bytes, RealMeanTime, ItemsMin)", "chart_y", "RealTime");
    mParser.addOption(chartYOption);
    
    QCommandLineOption chartZOption(QStringList() << "cz" << cz_name,
               "Chart Z-axis (e.g. auto, a2, t1)", "chart_z", "auto");
    mParser.addOption(chartZOption);
    
    QCommandLineOption appendOption(QStringList() << "ap" << fa_name,
               "Files to append by renaming (uses ';' as separator)", "files...");
    mParser.addOption(appendOption);
    
    QCommandLineOption overwriteOption(QStringList() << "ow" << fo_name,
               "Files to append by overwriting (uses ';' as separator)", "files...");
    mParser.addOption(overwriteOption);
}

bool CommandLineHandler::process(const QApplication& app)
{
    // Process
    mParser.process(app);
    
    const QStringList args = mParser.positionalArguments();
    
    if ( args.empty() )
        return false; // Not handled
    else if ( args.size() > 1)
        qWarning() << "[CmdLine] Ignoring additional arguments after first one";
    
    // Parse results
    QString errorMsg;
    BenchResults bchResults = ResultParser::parseJsonFile( args[0], errorMsg);
    
    if ( bchResults.benchmarks.isEmpty() ) {
        qCritical() << "[CmdLine] Error parsing file: " << args[0] << " -> " << errorMsg;
        return true;
    }
    
    // Get params
    QString chartType = mParser.value(ct_name).toLower();
    QString chartX    = mParser.value(cx_name).toLower();
    QString chartY    = mParser.value(cy_name).toLower();
    QString chartZ    = mParser.value(cz_name).toLower();
    QString apFiles   = mParser.value(fa_name);
    QString owFiles   = mParser.value(fo_name);
    
    //
    // Parse params
    PlotParams plotParams;
    
    // Append files
    bool multiFiles = false;
    QVector<FileReload> addFilenames;
    if ( !apFiles.isEmpty() )
    {
        QStringList apList = apFiles.split(';', Qt::SkipEmptyParts);
        for (const auto& fileName : qAsConst(apList))
        {
            if ( QFile::exists(fileName) )
            {
                QString errorMsg;
                BenchResults newResults = ResultParser::parseJsonFile(fileName, errorMsg);
                if (newResults.benchmarks.size() <= 0) {
                    qCritical() << "[CmdLine] Error parsing append file: " << fileName << " -> " << errorMsg;
                    return true;
                }
                bchResults.appendResults(newResults);
                multiFiles = true;
                addFilenames.append( {fileName, true} );
            }
        }
    }
    // Overwrite files
    if ( !owFiles.isEmpty() )
    {
        QStringList owList = owFiles.split(';', Qt::SkipEmptyParts);
        for (const auto& fileName : qAsConst(owList))
        {
            if ( QFile::exists(fileName) )
            {
                QString errorMsg;
                BenchResults newResults = ResultParser::parseJsonFile(fileName, errorMsg);
                if (newResults.benchmarks.size() <= 0) {
                    qCritical() << "[CmdLine] Error parsing overwrite file: " << fileName << " -> " << errorMsg;
                    return true;
                }
                bchResults.overwriteResults(newResults);
                multiFiles = true;
                addFilenames.append( {fileName, false} );
            }
        }
    }
    
    
    // Chart-type
    if      (chartType == "lines")      plotParams.type = ChartLineType;
    else if (chartType == "splines")    plotParams.type = ChartSplineType;
    else if (chartType == "bars")       plotParams.type = ChartBarType;
    else if (chartType == "hbars")      plotParams.type = ChartHBarType;
    else if (chartType == "boxes")      plotParams.type = ChartBoxType;
    else if (chartType == "3dbars")     plotParams.type = Chart3DBarsType;
    else if (chartType == "3dsurface")  plotParams.type = Chart3DSurfaceType;
    else {
        plotParams.type = ChartLineType;
        qWarning() << "[CmdLine] Unknown chart-type:" << chartType;
    }
    
    // X-axis
    if (chartX.size() >= 2 && (chartX.startsWith("a") || chartX.startsWith("t")))
    {
        if (chartX.startsWith("a")) plotParams.xType = PlotArgumentType;
        else                        plotParams.xType = PlotTemplateType;
        // Index
        chartX = chartX.mid(1);
        bool ok;
        int idx = chartX.toInt(&ok);
        if (ok && idx >= 1)
            plotParams.xIdx = idx - 1;
        else {
            plotParams.xIdx = 0;
            qWarning() << "[CmdLine] Unknown chart-x index:" << chartX;
        }
        // Invalid
        if (    (plotParams.xType == PlotArgumentType && plotParams.xIdx >= bchResults.meta.maxArguments)
             || (plotParams.xType == PlotTemplateType && plotParams.xIdx >= bchResults.meta.maxTemplates) )
        {
            if (    (plotParams.xType == PlotArgumentType && bchResults.meta.maxArguments == 0)
                 || (plotParams.xType == PlotTemplateType && bchResults.meta.maxTemplates == 0) )
            {
                plotParams.xType = PlotEmptyType;
                plotParams.xIdx = -1;
            }
            else
                plotParams.xIdx = 0;
            qWarning() << "[CmdLine] Chart-x index greater than number of parameters:" << chartX;
        }
    }
    else {
        plotParams.xType = PlotArgumentType;
        plotParams.xIdx = 0;
        qWarning() << "[CmdLine] Unknown chart-x:" << chartX;
    }
    
    // Y-axis
    if      (chartY == "cputime")    plotParams.yType = CpuTimeType;
    else if (chartY == "realtime")   plotParams.yType = RealTimeType;
    else if (chartY == "iterations") plotParams.yType = IterationsType;
    else if (chartY == "bytes" && bchResults.meta.hasBytesSec)  plotParams.yType = BytesType;
    else if (chartY == "items" && bchResults.meta.hasItemsSec)  plotParams.yType = ItemsType;
    
    else if (bchResults.meta.hasAggregate)
    {
        if      (chartY == "cpumintime")        plotParams.yType = CpuTimeMinType;
        else if (chartY == "cpumeantime")       plotParams.yType = CpuTimeMeanType;
        else if (chartY == "cpumediantime")     plotParams.yType = CpuTimeMedianType;
        else if (chartY == "realmintime")       plotParams.yType = RealTimeMinType;
        else if (chartY == "realmeantime")      plotParams.yType = RealTimeMeanType;
        else if (chartY == "realmediantime")    plotParams.yType = RealTimeMedianType;
        else if (bchResults.meta.hasBytesSec) {
            if      (chartY == "bytesmin")      plotParams.yType = BytesMinType;
            else if (chartY == "bytesmean")     plotParams.yType = BytesMeanType;
            else if (chartY == "bytesmedian")   plotParams.yType = BytesMedianType;
        }
        else if (bchResults.meta.hasItemsSec) {
            if      (chartY == "itemsmin")      plotParams.yType = ItemsMinType;
            else if (chartY == "itemsmean")     plotParams.yType = ItemsMeanType;
            else if (chartY == "itemsmedian")   plotParams.yType = ItemsMedianType;
        }
        else {
            plotParams.yType = RealTimeType;
            qWarning() << "[CmdLine] Unknown chart-y:" << chartY;
        }
    }
    else {
        plotParams.yType = RealTimeType;
        qWarning() << "[CmdLine] Unknown chart-y:" << chartY;
    }
    
    // Z-axis
    if (chartZ.size() >= 2
            && (chartZ == "auto" || chartZ.startsWith("a") || chartZ.startsWith("t")))
    {
        if (chartZ == "auto") {
            plotParams.zType = PlotEmptyType;
            plotParams.zIdx = 0;
        }
        else
        {
            if (chartZ.startsWith("a")) plotParams.zType = PlotArgumentType;
            else                        plotParams.zType = PlotTemplateType;
            // Index
            chartZ = chartZ.mid(1);
            bool ok;
            int idx = chartZ.toInt(&ok);
            if (ok && idx >= 1)
                plotParams.zIdx = idx - 1;
            else {
                plotParams.zIdx = 0;
                qWarning() << "[CmdLine] Unknown chart-z index:" << chartZ;
            }
            // Invalid
            if (    (plotParams.zType == PlotArgumentType && plotParams.zIdx >= bchResults.meta.maxArguments)
                 || (plotParams.zType == PlotTemplateType && plotParams.zIdx >= bchResults.meta.maxTemplates))
            {
                if (    (plotParams.zType == PlotArgumentType && bchResults.meta.maxArguments == 0)
                     || (plotParams.zType == PlotTemplateType && bchResults.meta.maxTemplates == 0) )
                {
                    plotParams.zType = PlotEmptyType;
                    plotParams.zIdx = -1;
                }
                else
                    plotParams.zIdx = 0;
                qWarning() << "[CmdLine] Chart-z index greater than number of parameters:" << chartZ;
            }
            else if (plotParams.zType == plotParams.xType && plotParams.zIdx == plotParams.xIdx) {
                qCritical() << "[CmdLine] Chart-z cannot be the same as chart-x";
                return true;
            }
        }
    }
    else {
        plotParams.zType = PlotEmptyType;
        plotParams.zIdx = 0;
        qWarning() << "[CmdLine] Unknown chart-z:" << chartZ;
    }
    
    
    //
    // Call plotter
    QFileInfo fileInfo( args[0] );
    QString fileName = fileInfo.fileName();
    if (multiFiles) fileName += " + ...";
    
    const auto& bchIdxs = bchResults.segmentAll();
    
    switch (plotParams.type)
    {
        case ChartLineType:
        case ChartSplineType:
        {
            PlotterLineChart *plotLines = new PlotterLineChart(bchResults, bchIdxs,
                                                               plotParams, fileName, addFilenames);
            plotLines->show();
            break;
        }
        case ChartBarType:
        case ChartHBarType:
        {
            PlotterBarChart *plotBars = new PlotterBarChart(bchResults, bchIdxs,
                                                             plotParams, fileName, addFilenames);
            plotBars->show();
            break;
        }
        case ChartBoxType:
        {
            PlotterBoxChart *plotBoxes = new PlotterBoxChart(bchResults, bchIdxs,
                                                             plotParams, fileName, addFilenames);
            plotBoxes->show();
            break;
        }
        case Chart3DBarsType:
        {
            Plotter3DBars *plot3DBars = new Plotter3DBars(bchResults, bchIdxs,
                                                          plotParams, fileName, addFilenames);
            plot3DBars->show();
            break;
        }
        case Chart3DSurfaceType:
        {
            Plotter3DSurface *plot3DSurface = new Plotter3DSurface(bchResults, bchIdxs,
                                                                   plotParams, fileName, addFilenames);
            plot3DSurface->show();
            break;
        }
    }
    
    // Handled
    return true;
}
