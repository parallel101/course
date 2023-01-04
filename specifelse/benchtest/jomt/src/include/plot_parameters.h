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

#ifndef PLOT_PARAMETERS_H
#define PLOT_PARAMETERS_H

#include "benchmark_results.h"

#include <QString>
#include <QVector>
#include <QStringList>

extern const char* config_folder;


// Chart types
enum PlotChartType {
    ChartLineType,
    ChartSplineType,
    ChartBarType,
    ChartHBarType,
    ChartBoxType,
    Chart3DBarsType,
    Chart3DSurfaceType
};

// Parameter types
enum PlotParamType {
    PlotEmptyType,
    PlotArgumentType,
    PlotTemplateType
};

// Y-value types
enum PlotValueType {
    CpuTimeType,  CpuTimeMinType,  CpuTimeMeanType,  CpuTimeMedianType,  CpuTimeStddevType,  CpuTimeCvType,
    RealTimeType, RealTimeMinType, RealTimeMeanType, RealTimeMedianType, RealTimeStddevType, RealTimeCvType,
    IterationsType,
    BytesType, BytesMinType, BytesMeanType, BytesMedianType, BytesStddevType, BytesCvType,
    ItemsType, ItemsMinType, ItemsMeanType, ItemsMedianType, ItemsStddevType, ItemsCvType
};

// Y-value stats
struct BenchYStats {
    double min, max;
    double median;
    double lowQuart, uppQuart;
};

// Plot parameters
struct PlotParams {
    PlotChartType type;
    PlotParamType xType;
    int xIdx;
    PlotValueType yType;
    PlotParamType zType;
    int zIdx;
};


/*
 * Helpers
 */
// Get Y-value according to type
double getYPlotValue(const BenchData &bchData, PlotValueType yType);

// Get Y-name according to type
QString getYPlotName(PlotValueType yType, QString timeUnit = "us");

// Convert time value to micro-seconds
double normalizeTimeUs(const BenchData &bchData, double value);

// Check Y-value type is time-based
bool isYTimeBased(PlotValueType yType);

// Find median in vector subpart
double findMedian(QVector<double> sorted, int begin, int end);

// Get Y-value statistics (for Box chart)
BenchYStats getYPlotStats(BenchData &bchData, PlotValueType yType);

// Compare first common elements of string lists
bool commonPartEqual(const QStringList &listA, const QStringList &listB);

// Check benchmark results have same origin files
bool sameResultsFiles(const QString &fileA, const QString &fileB,
                      const QVector<FileReload> &addFilesA, const QVector<FileReload> &addFilesB);


#endif // PLOT_PARAMETERS_H
