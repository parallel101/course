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

#include "plotter_3dsurface.h"
#include "ui_plotter_3dsurface.h"

#include "benchmark_results.h"
#include "result_parser.h"

#include <QFileInfo>
#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonObject>
#include <QJsonDocument>
#include <QtDataVisualization>

using namespace QtDataVisualization;

static const char* config_file = "config_3dsurface.json";
static const bool force_config = false;


Plotter3DSurface::Plotter3DSurface(const BenchResults &bchResults, const QVector<int> &bchIdxs,
                                   const PlotParams &plotParams, const QString &origFilename,
                                   const QVector<FileReload>& addFilenames, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Plotter3DSurface)
    , mBenchIdxs(bchIdxs)
    , mPlotParams(plotParams)
    , mOrigFilename(origFilename)
    , mAddFilenames(addFilenames)
    , mAllIndexes(bchIdxs.size() == bchResults.benchmarks.size())
    , mWatcher(parent)
{
    // UI
    ui->setupUi(this);
    this->setAttribute(Qt::WA_DeleteOnClose);
    
    QFileInfo fileInfo(origFilename);
    this->setWindowTitle("3D Surface - " + fileInfo.fileName());
    
    connectUI();
    
    // Init
    setupChart(bchResults, bchIdxs, plotParams);
    setupOptions();
    
    // Show
    QWidget *container = QWidget::createWindowContainer(mSurface);
    ui->horizontalLayout->insertWidget(0, container, 1);
}

Plotter3DSurface::~Plotter3DSurface()
{
    // Save options to file
    saveConfig();
    
    delete ui;
}

void Plotter3DSurface::connectUI()
{
    // Theme
    ui->comboBoxTheme->addItem("Primary Colors",    Q3DTheme::ThemePrimaryColors);
    ui->comboBoxTheme->addItem("Digia",             Q3DTheme::ThemeDigia);
    ui->comboBoxTheme->addItem("StoneMoss",         Q3DTheme::ThemeStoneMoss);
    ui->comboBoxTheme->addItem("ArmyBlue",          Q3DTheme::ThemeArmyBlue);
    ui->comboBoxTheme->addItem("Retro",             Q3DTheme::ThemeRetro);
    ui->comboBoxTheme->addItem("Ebony",             Q3DTheme::ThemeEbony);
    ui->comboBoxTheme->addItem("Isabelle",          Q3DTheme::ThemeIsabelle);
    ui->comboBoxTheme->addItem("Qt",                Q3DTheme::ThemeQt);
    connect(ui->comboBoxTheme, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DSurface::onComboThemeChanged);
    
    // Surface
    connect(ui->checkBoxFlip,  &QCheckBox::stateChanged, this, &Plotter3DSurface::onCheckFlip);
    
    setupGradients();
    connect(ui->comboBoxGradient, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DSurface::onComboGradientChanged);
    connect(ui->pushButtonSeries, &QPushButton::clicked, this, &Plotter3DSurface::onSeriesEditClicked);
    
    if (!isYTimeBased(mPlotParams.yType))
        ui->comboBoxTimeUnit->setEnabled(false);
    else
    {
        ui->comboBoxTimeUnit->addItem("ns", 1000.);
        ui->comboBoxTimeUnit->addItem("us", 1.);
        ui->comboBoxTimeUnit->addItem("ms", 0.001);
        connect(ui->comboBoxTimeUnit, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DSurface::onComboTimeUnitChanged);
    }
    
    // Axes
    ui->comboBoxAxis->addItem("X-Axis");
    ui->comboBoxAxis->addItem("Y-Axis");
    ui->comboBoxAxis->addItem("Z-Axis");
    connect(ui->comboBoxAxis, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DSurface::onComboAxisChanged);
    
    connect(ui->checkBoxAxisRotate,  &QCheckBox::stateChanged, this, &Plotter3DSurface::onCheckAxisRotate);
    connect(ui->checkBoxTitle,       &QCheckBox::stateChanged, this, &Plotter3DSurface::onCheckTitleVisible);
    connect(ui->checkBoxLog,         &QCheckBox::stateChanged, this, &Plotter3DSurface::onCheckLog);
    connect(ui->spinBoxLogBase,      QOverload<int>::of(&QSpinBox::valueChanged), this, &Plotter3DSurface::onSpinLogBaseChanged);
    connect(ui->lineEditTitle,       &QLineEdit::textChanged, this, &Plotter3DSurface::onEditTitleChanged);
    connect(ui->lineEditFormat,      &QLineEdit::textChanged, this, &Plotter3DSurface::onEditFormatChanged);
    connect(ui->doubleSpinBoxMin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DSurface::onSpinMinChanged);
    connect(ui->doubleSpinBoxMax,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DSurface::onSpinMaxChanged);
    connect(ui->spinBoxTicks,        QOverload<int>::of(&QSpinBox::valueChanged), this, &Plotter3DSurface::onSpinTicksChanged);
    connect(ui->spinBoxMTicks,       QOverload<int>::of(&QSpinBox::valueChanged), this, &Plotter3DSurface::onSpinMTicksChanged);
    
    // Actions
    connect(&mWatcher,              &QFileSystemWatcher::fileChanged, this, &Plotter3DSurface::onAutoReload);
    connect(ui->checkBoxAutoReload, &QCheckBox::stateChanged, this, &Plotter3DSurface::onCheckAutoReload);
    connect(ui->pushButtonReload,   &QPushButton::clicked, this, &Plotter3DSurface::onReloadClicked);
    connect(ui->pushButtonSnapshot, &QPushButton::clicked, this, &Plotter3DSurface::onSnapshotClicked);
}

void Plotter3DSurface::setupChart(const BenchResults &bchResults, const QVector<int> &bchIdxs, const PlotParams &plotParams, bool init)
{
    QScopedPointer<Q3DSurface> scopedSurface;
    Q3DSurface* surface = nullptr;
    if (init) {
        scopedSurface.reset( new Q3DSurface() );
        surface = scopedSurface.get();
    }
    else {  // Re-init
        surface = mSurface;
        const auto seriesList = surface->seriesList();
        for (const auto surfaceSeries : seriesList)
            surface->removeSeries(surfaceSeries);
        const auto surfaceAxes = surface->axes();
        for (const auto axis : surfaceAxes)
            surface->releaseAxis(axis);
        mSeriesMapping.clear();
    }
    Q_ASSERT(surface);
    
    // Time unit
    mCurrentTimeFactor = 1.;
    if ( isYTimeBased(mPlotParams.yType) ) {
        if (     bchResults.meta.time_unit == "ns") mCurrentTimeFactor = 1000.;
        else if (bchResults.meta.time_unit == "ms") mCurrentTimeFactor = 0.001;
    }
    
    
    // 3D
    // X: argumentA or templateB
    // Y: time/iter/bytes/items (not name dependent)
    // Z: argumentC or templateD (with C!=A, D!=B)
    bool custXAxis = true, custZAxis = true;
    QString custXName, custZName;
    bool hasZParam = plotParams.zType != PlotEmptyType;
    
    //
    // No Z-param -> one row per benchmark type
    if (!hasZParam)
    {
        // Single series (i.e. color)
        QSurfaceDataProxy *dataProxy = new QSurfaceDataProxy();
        QScopedPointer<QSurface3DSeries> series(new QSurface3DSeries(dataProxy));
        QScopedPointer<QSurfaceDataArray> dataArray(new QSurfaceDataArray);
        
        // Segment per X-param
        QVector<BenchSubset> bchSubsets = bchResults.groupParam(plotParams.xType == PlotArgumentType,
                                                                bchIdxs, plotParams.xIdx, "X");
        // Check subsets symmetry/min size
        bool symBchOK = true, symOK = true, minOK = true;
        QString culpritName;
        int refSize = bchSubsets.empty() ? 0 : bchSubsets[0].idxs.size();
        for (int i = 0; symOK && minOK && i < bchSubsets.size(); ++i) {
            symOK = bchSubsets[i].idxs.size() == refSize;
            minOK = bchSubsets[i].idxs.size() >= 2;
            if (!symOK || !minOK)
                culpritName = bchSubsets[i].name;
        }
        // Ignore asymmetrical series
        if (!symOK) {
            qWarning() << "Inconsistent number of X-values between benchmarks to trace surface for: " << culpritName;
        }
        // Ignore single-row series
        else if (!minOK) {
            qWarning() << "Not enough X-values to trace surface for: " << culpritName;
        }
        else
        {
            int prevRowSize = 0;
            double zFallback = 0.;
            for (const auto& bchSubset : qAsConst(bchSubsets))
            {
                // Check inter benchmark consistency
                if (prevRowSize > 0 && prevRowSize != bchSubset.idxs.size()) {
                    symBchOK = false;
                    qWarning() << "Inconsistent number of X-values between benchmarks to trace surface";
                    break;
                }
                prevRowSize = bchSubset.idxs.size();
                
                // One row per X-group
                QScopedPointer<QSurfaceDataRow> newRow(new QSurfaceDataRow( bchSubset.idxs.size() ));
                
//                const QString & subsetName = bchSubset.name;
//                qDebug() << "subsetName:" << subsetName;
//                qDebug() << "subsetIdxs:" << bchSubset.idxs;
                
                int index = 0;
                double xFallback = 0.;
                for (int idx : bchSubset.idxs)
                {
                    QString xName = bchResults.getParamName(plotParams.xType == PlotArgumentType,
                                                            idx, plotParams.xIdx);
                    double xVal = BenchResults::getParamValue(xName, custXName, custXAxis, xFallback);
                    
                    // Y val
                    double yVal = getYPlotValue(bchResults.benchmarks[idx], plotParams.yType) * mCurrentTimeFactor;
//                    qDebug() << "-> [" << xVal << yVal << zFallback << "]";
                    
                    // Add column
                    (*newRow)[index++].setPosition( QVector3D(xVal, yVal, zFallback) );
                }
                // Add row
                dataArray->append(newRow.take());
                
                ++zFallback;
            }
        }
        if (symBchOK && dataArray->size() > 0)
        {
            // Add series
            dataProxy->resetArray(dataArray.take());
            
            series->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
            series->setFlatShadingEnabled(true);
            series->setItemLabelFormat(QStringLiteral("[@xLabel, @zLabel]: @yLabel"));
            mSeriesMapping.push_back({"", ""}); // color set later
            
            surface->addSeries(series.take());
        }
    }
    //
    // Z-param -> one series per benchmark type
    else
    {
        // Initial segmentation by 'full name % param1 % param2' (group benchmarks)
        const auto bchNames = bchResults.segment2DNames(bchIdxs,
                                                        plotParams.xType == PlotArgumentType, plotParams.xIdx,
                                                        plotParams.zType == PlotArgumentType, plotParams.zIdx);
        for (const auto& bchName : bchNames)
        {
            // One series (i.e. color) per 2D-name
            QSurfaceDataProxy *dataProxy = new QSurfaceDataProxy();
            QScopedPointer<QSurface3DSeries> series(new QSurface3DSeries(dataProxy));
            
//            qDebug() << "bchName" << bchName.name << "|" << bchName.idxs;
            
            // One subset per Z-param from 2D-names
            QVector<BenchSubset> bchZSubs = bchResults.segmentParam(plotParams.zType == PlotArgumentType,
                                                                    bchName.idxs, plotParams.zIdx);
            // Ignore incompatible series
            if ( bchZSubs.isEmpty() ) {
                qWarning() << "No Z-value to trace surface for other benchmarks";
                continue;
            }
            
            // Check subsets symmetry/min size
            bool symOK = true, minOK = true;
            QString culpritName;
            int refSize = bchZSubs[0].idxs.size();
            for (int i=0; symOK && minOK && i<bchZSubs.size(); ++i) {
                symOK = bchZSubs[i].idxs.size() == refSize;
                minOK = bchZSubs[i].idxs.size() >= 2;
                if (!symOK || !minOK)
                    culpritName = bchZSubs[0].name;
            }
            // Ignore asymmetrical series
            if (!symOK) {
                qWarning() << "Inconsistent number of X-values between benchmarks to trace surface for: "
                           << bchName.name + " [Z=" + culpritName + "]";
                continue;
            }
            // Ignore single-row series
            else if (!minOK) {
                qWarning() << "Not enough X-values to trace surface for: "
                           << bchName.name + " [Z=" + culpritName + "]";
                continue;
            }

            QScopedPointer<QSurfaceDataArray> dataArray(new QSurfaceDataArray);
            double zFallback = 0.;
            for (const auto& bchZSub : qAsConst(bchZSubs))
            {
                QString zName = bchZSub.name;
//                qDebug() << "bchZSub" << bchZSub.name << "|" << bchZSub.idxs;

                double zVal = BenchResults::getParamValue(zName, custZName, custZAxis, zFallback);
                
                // One row per Z-param from 2D-names
                QScopedPointer<QSurfaceDataRow> newRow(new QSurfaceDataRow( bchZSub.idxs.size() ));
                
                // One subset per X-param from Z-Subset
                QVector<BenchSubset> bchSubsets = bchResults.groupParam(plotParams.xType == PlotArgumentType,
                                                                        bchZSub.idxs, plotParams.xIdx, "X");
                Q_ASSERT(bchSubsets.size() <= 1);
                for (const auto& bchSubset : qAsConst(bchSubsets))
                {
                    int index = 0;
                    double xFallback = 0.;
                    for (int idx : bchSubset.idxs)
                    {
                        QString xName = bchResults.getParamName(plotParams.xType == PlotArgumentType,
                                                                idx, plotParams.xIdx);
                        double xVal = BenchResults::getParamValue(xName, custXName, custXAxis, xFallback);
                        
                        // Y val
                        double yVal = getYPlotValue(bchResults.benchmarks[idx], plotParams.yType) * mCurrentTimeFactor;
//                        qDebug() << "-> [" << xVal << yVal << zVal << "]";
                        
                        // Add column
                        (*newRow)[index++].setPosition( QVector3D(xVal, yVal, zVal) );
                    }
                    // Add row
                    dataArray->append(newRow.take());
                }
            }
            // Add series
            dataProxy->resetArray(dataArray.take());
            
            series->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
            series->setFlatShadingEnabled(true);
            series->setName(bchName.name);
            mSeriesMapping.push_back({bchName.name, bchName.name}); // color set later
            series->setItemLabelFormat(QStringLiteral("@seriesName [@xLabel, @zLabel]: @yLabel"));
            
            surface->addSeries(series.take());
        }
    }
    
    // Axes
    if ( !surface->seriesList().isEmpty() && surface->seriesList().constFirst()->dataProxy()->rowCount() > 0)
    {
        // General
        surface->setHorizontalAspectRatio(1.0);
        surface->setShadowQuality(QAbstract3DGraph::ShadowQualitySoftMedium);
        
        // X-axis
        QValue3DAxis *xAxis = surface->axisX();
        if (plotParams.xType == PlotArgumentType)
            xAxis->setTitle("Argument " + QString::number(plotParams.xIdx+1));
        else { // template
            if ( !custXName.isEmpty() )
                xAxis->setTitle(custXName);
            else
                xAxis->setTitle("Template " + QString::number(plotParams.xIdx+1));
        }
        xAxis->setTitleVisible(true);
        xAxis->setSegmentCount(8);
        
        // Y-axis
        QValue3DAxis *yAxis = surface->axisY();
        yAxis->setTitle( getYPlotName(plotParams.yType, bchResults.meta.time_unit) );
        yAxis->setTitleVisible(true);
        
        // Z-axis
        QValue3DAxis *zAxis = surface->axisZ();
        if (plotParams.zType != PlotEmptyType)
        {
            if (plotParams.zType == PlotArgumentType)
                zAxis->setTitle("Argument " + QString::number(plotParams.zIdx+1));
            else { // template
                if ( !custZName.isEmpty() )
                    zAxis->setTitle(custZName);
                else
                    zAxis->setTitle("Template " + QString::number(plotParams.zIdx+1));
            }
            zAxis->setTitleVisible(true);
        }
        zAxis->setSegmentCount(8);
    }
    else {
        // Title-like
        QValue3DAxis *yAxis = surface->axisY();
        yAxis->setTitle("No compatible series to display");
        yAxis->setTitleVisible(true);

        qWarning() << "No compatible series to display";
    }
    
    if (init)
    {
        // Take
        mSurface = scopedSurface.take();
    }
}

void Plotter3DSurface::setupOptions(bool init)
{
    // General
    if (init) {
        mSurface->activeTheme()->setType(Q3DTheme::ThemePrimaryColors);
    }
    
    mIgnoreEvents = true;
    int prevAxisIdx = ui->comboBoxAxis->currentIndex();
    
    if (!init)  // Re-init
    {
        ui->comboBoxAxis->setCurrentIndex(0);
        for (auto &axisParams : mAxesParams)
            axisParams.reset();
        ui->checkBoxAxisRotate->setChecked(false);
        ui->checkBoxTitle->setChecked(true);
        ui->checkBoxLog->setChecked(false);
        ui->spinBoxLogBase->setValue(10);
        ui->comboBoxGradient->setCurrentIndex(0);
    }
    
    // Time unit
    if      (mCurrentTimeFactor > 1.) ui->comboBoxTimeUnit->setCurrentIndex(0); // ns
    else if (mCurrentTimeFactor < 1.) ui->comboBoxTimeUnit->setCurrentIndex(2); // ms
    else                              ui->comboBoxTimeUnit->setCurrentIndex(1); // us
    
    // Axes
    // X-axis
    QValue3DAxis *xAxis = mSurface->axisX();
    if (xAxis)
    {
        auto& axisParam = mAxesParams[0];
        
        axisParam.titleText = xAxis->title();
        axisParam.title = !axisParam.titleText.isEmpty();
        axisParam.labelFormat = "%g";
        xAxis->setLabelFormat(axisParam.labelFormat);
        axisParam.min = xAxis->min();
        axisParam.max = xAxis->max();
        axisParam.ticks = xAxis->segmentCount();
        axisParam.mticks = xAxis->subSegmentCount();
        
        ui->checkBoxTitle->setChecked( axisParam.title );
        ui->lineEditTitle->setText( axisParam.titleText );
        ui->lineEditTitle->setCursorPosition(0);
        ui->lineEditFormat->setText( axisParam.labelFormat );
        ui->lineEditFormat->setCursorPosition(0);
        ui->doubleSpinBoxMin->setValue( axisParam.min );
        ui->doubleSpinBoxMax->setValue( axisParam.max );
        ui->spinBoxTicks->setValue( axisParam.ticks );
        ui->spinBoxMTicks->setValue( axisParam.mticks );
    }
    // Y-axis
    QValue3DAxis *yAxis = mSurface->axisY();
    if (yAxis)
    {
        auto& axisParam = mAxesParams[1];
        
        axisParam.titleText = yAxis->title();
        axisParam.title = !axisParam.titleText.isEmpty();
        axisParam.labelFormat = yAxis->labelFormat();
        axisParam.min = yAxis->min();
        axisParam.max = yAxis->max();
        axisParam.ticks = yAxis->segmentCount();
        axisParam.mticks = yAxis->subSegmentCount();
    }
    // Z-axis
    QValue3DAxis *zAxis = mSurface->axisZ();
    if (zAxis)
    {
        auto& axisParam = mAxesParams[2];
        
        axisParam.titleText = zAxis->title();
        axisParam.title = !axisParam.titleText.isEmpty();
        axisParam.labelFormat = "%g";
        zAxis->setLabelFormat(axisParam.labelFormat);
        axisParam.min = zAxis->min();
        axisParam.max = zAxis->max();
        axisParam.ticks = zAxis->segmentCount();
        axisParam.mticks = zAxis->subSegmentCount();
    }
    mIgnoreEvents = false;
    
    
    // Load options from file
    loadConfig(init);
    
    
    // Apply actions
    if (ui->checkBoxAutoReload->isChecked())
        onCheckAutoReload(Qt::Checked);
    
    // Update series color config
    const auto& chartSeries = mSurface->seriesList();
    for (int idx = 0 ; idx < mSeriesMapping.size(); ++idx)
    {
        auto& config = mSeriesMapping[idx];
        const auto& series = chartSeries.at(idx);
        
        config.oldColor = series->baseColor();
        if (!config.newColor.isValid())
            config.newColor = series->baseColor();  // init
        else
            series->setBaseColor(config.newColor);  // apply
        
        if (config.newName != config.oldName)
            series->setName( config.newName );
    }
    
    // Restore selected axis
    if (!init)
        ui->comboBoxAxis->setCurrentIndex(prevAxisIdx);
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString()+ ")");
}

void Plotter3DSurface::loadConfig(bool init)
{
    QFile configFile(QString(config_folder) + config_file);
    if (configFile.open(QIODevice::ReadOnly))
    {
        QByteArray configData = configFile.readAll();
        configFile.close();
        QJsonDocument configDoc(QJsonDocument::fromJson(configData));
        QJsonObject json = configDoc.object();
        
        // Theme
        if (json.contains("theme") && json["theme"].isString())
            ui->comboBoxTheme->setCurrentText( json["theme"].toString() );
        
        // Surface
        if (json.contains("surface.flip") && json["surface.flip"].isBool())
            ui->checkBoxFlip->setChecked( json["surface.flip"].toBool() );
        if (json.contains("surface.gradient") && json["surface.gradient"].isString())
            ui->comboBoxGradient->setCurrentText( json["surface.gradient"].toString() );
        
        // Series
        if (json.contains("series") && json["series"].isArray())
        {
            auto series = json["series"].toArray();
            for (int idx = 0; idx < series.size(); ++idx) {
                QJsonObject config = series[idx].toObject();
                if ( config.contains("oldName")  && config["oldName"].isString()
                  && config.contains("newName")  && config["newName"].isString()
                  && config.contains("newColor") && config["newColor"].isString()
                  && QColor::isValidColor(config["newColor"].toString()) )
                {
                    SeriesConfig savedConfig(config["oldName"].toString(), "");
                    int iCfg = mSeriesMapping.indexOf(savedConfig);
                    if (iCfg >= 0) {
                        mSeriesMapping[iCfg].newName = config["newName"].toString();
                        mSeriesMapping[iCfg].newColor.setNamedColor( config["newColor"].toString() );
                    }
                }
            }
        }
        
        // Time
        if (!init) {
            if (json.contains("timeUnit") && json["timeUnit"].isString())
                ui->comboBoxTimeUnit->setCurrentText( json["timeUnit"].toString() );
        }
        
        // Actions
        if (json.contains("autoReload") && json["autoReload"].isBool())
            ui->checkBoxAutoReload->setChecked( json["autoReload"].toBool() );
        
        // Axes
        QString prefix = "axis.x";
        for (int idx = 0; idx < 3; ++idx)
        {
            auto& axis = mAxesParams[idx];
            
            if (json.contains(prefix + ".rotate") && json[prefix + ".rotate"].isBool()) {
                axis.rotate = json[prefix + ".rotate"].toBool();
                ui->checkBoxAxisRotate->setChecked( axis.rotate );
            }
            if (json.contains(prefix + ".title") && json[prefix + ".title"].isBool()) {
                axis.title = json[prefix + ".title"].toBool();
                ui->checkBoxTitle->setChecked( axis.title );
            }
            if (json.contains(prefix + ".log") && json[prefix + ".log"].isBool()) {
                axis.log = json[prefix + ".log"].toBool();
                ui->checkBoxLog->setChecked( axis.log );
            }
            if (json.contains(prefix + ".logBase") && json[prefix + ".logBase"].isDouble()) {
                axis.logBase = json[prefix + ".logBase"].toInt(10);
                ui->spinBoxLogBase->setValue( axis.logBase );
            }
            if (json.contains(prefix + ".labelFormat") && json[prefix + ".labelFormat"].isString()) {
                axis.labelFormat = json[prefix + ".labelFormat"].toString();
                ui->lineEditFormat->setText( axis.labelFormat );
                ui->lineEditFormat->setCursorPosition(0);
            }
            if (json.contains(prefix + ".ticks") && json[prefix + ".ticks"].isDouble()) {
                axis.ticks = json[prefix + ".ticks"].toInt(idx == 1 ? 5 : 8);
                ui->spinBoxTicks->setValue( axis.ticks );
            }
            if (json.contains(prefix + ".mticks") && json[prefix + ".mticks"].isDouble()) {
                axis.mticks = json[prefix + ".mticks"].toInt(1);
                ui->spinBoxMTicks->setValue( axis.mticks );
            }
            if (!init)
            {
                if (json.contains(prefix + ".titleText") && json[prefix + ".titleText"].isString()) {
                    axis.titleText = json[prefix + ".titleText"].toString();
                    ui->lineEditTitle->setText( axis.titleText );
                    ui->lineEditTitle->setCursorPosition(0);
                }
                if (idx == 1 || force_config)
                {
                    if (json.contains(prefix + ".min") && json[prefix + ".min"].isDouble()) {
                        axis.min = json[prefix + ".min"].toDouble();
                        ui->doubleSpinBoxMin->setValue( axis.min );
                    }
                    if (json.contains(prefix + ".max") && json[prefix + ".max"].isDouble()) {
                        axis.max = json[prefix + ".max"].toDouble();
                        ui->doubleSpinBoxMax->setValue( axis.max );
                    }
                }
            }
            if (idx == 0)
            {
                prefix = "axis.y";
                ui->comboBoxAxis->setCurrentIndex(1);
            }
            else if (idx == 1)
            {
                prefix = "axis.z";
                ui->comboBoxAxis->setCurrentIndex(2);
            }
        }
        ui->comboBoxAxis->setCurrentIndex(0);
    }
    else
    {
        if (configFile.exists())
            qWarning() << "Couldn't read: " << QString(config_folder) + config_file;
    }
}

void Plotter3DSurface::saveConfig()
{
    QFile configFile(QString(config_folder) + config_file);
    if (configFile.open(QIODevice::WriteOnly))
    {
        QJsonObject json;
        
        // Theme
        json["theme"] = ui->comboBoxTheme->currentText();
        // Surface
        json["surface.flip"]     = ui->checkBoxFlip->isChecked();
        json["surface.gradient"] = ui->comboBoxGradient->currentText();
        // Series
        QJsonArray series;
        for (const auto& seriesConfig : qAsConst(mSeriesMapping)) {
            QJsonObject config;
            config["oldName"] = seriesConfig.oldName;
            config["newName"] = seriesConfig.newName;
            config["newColor"] = seriesConfig.newColor.name();
            series.append(config);
        }
        if (!series.empty())
            json["series"] = series;
        // Time
        json["timeUnit"] = ui->comboBoxTimeUnit->currentText();
        // Actions
        json["autoReload"] = ui->checkBoxAutoReload->isChecked();
        // Axes
        QString prefix = "axis.x";
        for (int idx = 0; idx < 3; ++idx)
        {
            const auto& axis = mAxesParams[idx];
            
            json[prefix + ".rotate"]      = axis.rotate;
            json[prefix + ".title"]       = axis.title;
            json[prefix + ".log"]         = axis.log;
            json[prefix + ".logBase"]     = axis.logBase;
            json[prefix + ".titleText"]   = axis.titleText;
            json[prefix + ".labelFormat"] = axis.labelFormat;
            json[prefix + ".min"]         = axis.min;
            json[prefix + ".max"]         = axis.max;
            json[prefix + ".ticks"]       = axis.ticks;
            json[prefix + ".mticks"]      = axis.mticks;
            
            if (idx == 0)
                prefix = "axis.y";
            else if (idx == 1)
                prefix = "axis.z";
        }
        
        configFile.write( QJsonDocument(json).toJson() );
    }
    else
        qWarning() << "Couldn't update: " << QString(config_folder) + config_file;
}

void Plotter3DSurface::setupGradients()
{
    ui->comboBoxGradient->addItem("No gradient");
    
    ui->comboBoxGradient->addItem("Deep volcano"); {
        QLinearGradient gr;
        gr.setColorAt(0.0,  Qt::black); gr.setColorAt(0.33, Qt::blue);
        gr.setColorAt(0.67, Qt::red);   gr.setColorAt(1.0,  Qt::yellow);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Jungle heat"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::darkGreen); gr.setColorAt(0.5, Qt::yellow);
        gr.setColorAt(0.8, Qt::red);       gr.setColorAt(1.0, Qt::darkRed);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Spectral redux"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::blue);   gr.setColorAt(0.33, Qt::green);
        gr.setColorAt(0.5, Qt::yellow); gr.setColorAt(1.0,  Qt::red);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Spectral extended"); {
        QLinearGradient gr;
        gr.setColorAt(0.0,  Qt::magenta); gr.setColorAt(0.25, Qt::blue);
        gr.setColorAt(0.5,  Qt::cyan);
        gr.setColorAt(0.67, Qt::green);   gr.setColorAt(0.83, Qt::yellow);
        gr.setColorAt(1.0,  Qt::red);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Reddish"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::darkRed); gr.setColorAt(1.0, Qt::red);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Greenish"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::darkGreen); gr.setColorAt(1.0, Qt::green);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Bluish"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::darkCyan); gr.setColorAt(1.0, Qt::cyan);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Gray"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::black); gr.setColorAt(1.0, Qt::white);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Gray inverted"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::white); gr.setColorAt(1.0, Qt::black);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Gray centered"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::black); gr.setColorAt(0.5, Qt::white);
        gr.setColorAt(1.0, Qt::black);
        mGrads.push_back(gr);
    }
    
    ui->comboBoxGradient->addItem("Gray inv-centered"); {
        QLinearGradient gr;
        gr.setColorAt(0.0, Qt::white); gr.setColorAt(0.5, Qt::black);
        gr.setColorAt(1.0, Qt::white);
        mGrads.push_back(gr);
    }
}

//
// Theme
void Plotter3DSurface::onComboThemeChanged(int index)
{
    Q3DTheme::Theme theme = static_cast<Q3DTheme::Theme>(
                ui->comboBoxTheme->itemData(index).toInt());
    mSurface->activeTheme()->setType(theme);
    
    onComboGradientChanged( ui->comboBoxGradient->currentIndex() );
    
    // Update series color
    const auto& chartSeries = mSurface->seriesList();
    for (int idx = 0 ; idx < mSeriesMapping.size(); ++idx)
    {
        auto& config = mSeriesMapping[idx];
        const auto& series = chartSeries.at(idx);
        auto prevColor = config.oldColor;
        
        config.oldColor = series->baseColor();
        if (config.newColor != prevColor)
            series->setBaseColor(config.newColor);  // re-apply config
        else
            config.newColor = config.oldColor;      // sync with theme
    }
}

//
// Surface
void Plotter3DSurface::onCheckFlip(int state)
{
    mSurface->setFlipHorizontalGrid(state == Qt::Checked);
}

void Plotter3DSurface::onComboGradientChanged(int idx)
{
    if (idx == 0)
    {
        for (auto& series : mSurface->seriesList())
            series->setColorStyle(Q3DTheme::ColorStyleUniform);
    }
    else
    {
        for (auto& series : mSurface->seriesList()) {
            series->setBaseGradient( mGrads[idx-1] );
            series->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
        }
    }
}

void Plotter3DSurface::onSeriesEditClicked()
{
    SeriesDialog seriesDialog(mSeriesMapping, this);
    auto res = seriesDialog.exec();
    if (res == QDialog::Accepted)
    {
        const auto& chartSeries = mSurface->seriesList();
        const auto& newMapping = seriesDialog.getMapping();
        for (int idx = 0; idx < newMapping.size(); ++idx)
        {
            const auto& newPair = newMapping[idx];
            const auto& oldPair = mSeriesMapping[idx];
            auto series = chartSeries.at(idx);
            if (newPair.newName != oldPair.newName) {
                series->setName( newPair.newName );
            }
            if (newPair.newColor != oldPair.newColor) {
                series->setBaseColor(newPair.newColor);
            }
        }
        mSeriesMapping = newMapping;
    }
}

void Plotter3DSurface::onComboTimeUnitChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    
    // Update data
    double unitFactor = ui->comboBoxTimeUnit->currentData().toDouble();
    double updateFactor = unitFactor / mCurrentTimeFactor;  // can cause precision loss
    auto chartSeries = mSurface->seriesList();
    if (chartSeries.empty())
        return;
    
    for (auto& series : chartSeries)
    {
        const auto& dataProxy = series->dataProxy();
        for (int iR = 0; iR < dataProxy->rowCount(); ++iR)
        {
            for (int iC = 0; iC < dataProxy->columnCount(); ++iC)
            {
                auto item = dataProxy->itemAt(iR, iC);
                dataProxy->setItem(iR, iC,
                                   QSurfaceDataItem( QVector3D(item->x(), item->y() * updateFactor, item->z()) ));
            }
        }
    }
    
    // Update axis title
    QString oldUnitName = "(us)";
    if      (mCurrentTimeFactor > 1.) oldUnitName = "(ns)";
    else if (mCurrentTimeFactor < 1.) oldUnitName = "(ms)";
    
    auto yAxis = mSurface->axisY();
    if (yAxis) {
        QString axisTitle = yAxis->title();
        if (axisTitle.endsWith(oldUnitName)) {
            QString unitName  = ui->comboBoxTimeUnit->currentText();
            onEditTitleChanged2(axisTitle.replace(axisTitle.size() - 3, 2, unitName), 1);
        }
    }
    // Update range
    if (ui->comboBoxAxis->currentIndex() == 1)
    {
        if (updateFactor > 1.) {    // enforce proper order
            ui->doubleSpinBoxMax->setValue(mAxesParams[1].max * updateFactor);
            ui->doubleSpinBoxMin->setValue(mAxesParams[1].min * updateFactor);
        }
        else {
            ui->doubleSpinBoxMin->setValue(mAxesParams[1].min * updateFactor);
            ui->doubleSpinBoxMax->setValue(mAxesParams[1].max * updateFactor);
        }
    }
    else {
        if (updateFactor > 1.) {    // enforce proper order
            onSpinMaxChanged2(mAxesParams[1].max * updateFactor, 1);
            onSpinMinChanged2(mAxesParams[1].min * updateFactor, 1);
        }
        else {
            onSpinMinChanged2(mAxesParams[1].min * updateFactor, 1);
            onSpinMaxChanged2(mAxesParams[1].max * updateFactor, 1);
        }
    }
    
    mCurrentTimeFactor = unitFactor;
}

//
// Axes
void Plotter3DSurface::onComboAxisChanged(int idx)
{
    // Update UI
    bool wasIgnoring = mIgnoreEvents;
    mIgnoreEvents = true;
    
    ui->checkBoxAxisRotate->setChecked( mAxesParams[idx].rotate );
    ui->checkBoxTitle->setChecked( mAxesParams[idx].title );
    ui->checkBoxLog->setChecked( mAxesParams[idx].log );
    ui->spinBoxLogBase->setValue( mAxesParams[idx].logBase );
    ui->spinBoxLogBase->setEnabled( ui->checkBoxLog->isChecked() );
    ui->lineEditTitle->setText( mAxesParams[idx].titleText );
    ui->lineEditTitle->setCursorPosition(0);
    ui->lineEditFormat->setText( mAxesParams[idx].labelFormat );
    ui->lineEditFormat->setCursorPosition(0);
    ui->doubleSpinBoxMin->setDecimals(idx == 1 ? 6 : 3);
    ui->doubleSpinBoxMax->setDecimals(idx == 1 ? 6 : 3);
    ui->doubleSpinBoxMin->setValue( mAxesParams[idx].min );
    ui->doubleSpinBoxMax->setValue( mAxesParams[idx].max );
    ui->doubleSpinBoxMin->setSingleStep(idx == 1 ? 0.1 : 1.0);
    ui->doubleSpinBoxMax->setSingleStep(idx == 1 ? 0.1 : 1.0);
    ui->spinBoxTicks->setValue( mAxesParams[idx].ticks );
    ui->spinBoxTicks->setEnabled( !ui->checkBoxLog->isChecked() );
    ui->spinBoxMTicks->setValue( mAxesParams[idx].mticks );
    ui->spinBoxMTicks->setEnabled( !ui->checkBoxLog->isChecked() );
    
    mIgnoreEvents = wasIgnoring;
}

void Plotter3DSurface::onCheckAxisRotate(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setTitleFixed(state != Qt::Checked);
        axis->setLabelAutoRotation(state == Qt::Checked ? 90 : 0);
        mAxesParams[iAxis].rotate = state == Qt::Checked;
    }
}

void Plotter3DSurface::onCheckTitleVisible(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setTitleVisible(state == Qt::Checked);
        mAxesParams[iAxis].title = state == Qt::Checked;
    }
}

void Plotter3DSurface::onCheckLog(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis)
    {
        if (state == Qt::Checked) {
            axis->setFormatter(new QLogValue3DAxisFormatter());
            ui->doubleSpinBoxMin->setMinimum( 0.001 );
            mAxesParams[iAxis].min = axis->min();
        }
        else {
            axis->setFormatter(new QValue3DAxisFormatter());
            ui->doubleSpinBoxMin->setMinimum( 0. );
            mAxesParams[iAxis].max = axis->max();
        }
        mAxesParams[iAxis].log = state == Qt::Checked;
        ui->spinBoxTicks->setEnabled(  state != Qt::Checked);
        ui->spinBoxMTicks->setEnabled( state != Qt::Checked);
        ui->spinBoxLogBase->setEnabled(state == Qt::Checked);
    }
}

void Plotter3DSurface::onSpinLogBaseChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis)
    {
        QLogValue3DAxisFormatter* formatter = (QLogValue3DAxisFormatter*)axis->formatter();
        if (formatter) {
            formatter->setBase(i);
            mAxesParams[iAxis].logBase = i;
        }
    }
}

void Plotter3DSurface::onEditTitleChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onEditTitleChanged2(text, iAxis);
}

void Plotter3DSurface::onEditTitleChanged2(const QString& text, int iAxis)
{
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setTitle(text);
        mAxesParams[iAxis].titleText = text;
    }
}

void Plotter3DSurface::onEditFormatChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setLabelFormat(text);
        mAxesParams[iAxis].labelFormat = text;
    }
}

void Plotter3DSurface::onSpinMinChanged(double d)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinMinChanged2(d, iAxis);
}

void Plotter3DSurface::onSpinMinChanged2(double d, int iAxis)
{
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setMin(d);
        mAxesParams[iAxis].min = d;
    }
}

void Plotter3DSurface::onSpinMaxChanged(double d)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinMaxChanged2(d, iAxis);
}

void Plotter3DSurface::onSpinMaxChanged2(double d, int iAxis)
{
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setMax(d);
        mAxesParams[iAxis].max = d;
    }
}

void Plotter3DSurface::onSpinTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setSegmentCount(i);
        mAxesParams[iAxis].ticks = i;
    }
}

void Plotter3DSurface::onSpinMTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QValue3DAxis* axis;
    if      (iAxis == 0)    axis = mSurface->axisX();
    else if (iAxis == 1)    axis = mSurface->axisY();
    else                    axis = mSurface->axisZ();
    
    if (axis) {
        axis->setSubSegmentCount(i);
        mAxesParams[iAxis].mticks = i;
    }
}

//
// Actions
void Plotter3DSurface::onCheckAutoReload(int state)
{
    if (state == Qt::Checked)
    {
        if (mWatcher.files().empty())
        {
            mWatcher.addPath(mOrigFilename);
            for (const auto& addFilename : mAddFilenames)
                mWatcher.addPath( addFilename.filename );
        }
    }
    else
    {
        if (!mWatcher.files().empty())
            mWatcher.removePaths( mWatcher.files() );
    }
}

void Plotter3DSurface::onAutoReload(const QString &path)
{
    QFileInfo fi(path);
    if (fi.exists() && fi.isReadable() && fi.size() > 0)
        onReloadClicked();
    else
        qWarning() << "Unable to auto-reload file: " << path;
}

void Plotter3DSurface::onReloadClicked()
{
    // Load new results
    QString errorMsg;
    BenchResults newBchResults = ResultParser::parseJsonFile( mOrigFilename, errorMsg );
    
    if ( newBchResults.benchmarks.isEmpty() ) {
        QMessageBox::critical(this, "Chart reload", "Error parsing original file: " + mOrigFilename + " -> " + errorMsg);
        return;
    }
    
    for (const auto& addFile : qAsConst(mAddFilenames))
    {
        errorMsg.clear();
        BenchResults newAddResults = ResultParser::parseJsonFile(addFile.filename, errorMsg);
        if ( newAddResults.benchmarks.isEmpty() ) {
            QMessageBox::critical(this, "Chart reload", "Error parsing additional file: " + addFile.filename + " -> " + errorMsg);
            return;
        }
        
        if (addFile.isAppend)
            newBchResults.appendResults(newAddResults);
        else
            newBchResults.overwriteResults(newAddResults);
    }
    
    // Check compatibility with previous
    errorMsg.clear();
    if (mBenchIdxs.size() != newBchResults.benchmarks.size())
    {
        errorMsg = "Number of series/points is different";
        if (mAllIndexes)
        {
            mBenchIdxs.clear();
            for (int i=0; i<newBchResults.benchmarks.size(); ++i)
                mBenchIdxs.append(i);
        }
    }
    
    while ( errorMsg.isEmpty() )  // once
    {
        // Check chart type
        bool hasZParam = mPlotParams.zType != PlotEmptyType;
        if (!hasZParam)
        {
            // Check compatibility with previous
            const auto& oldSurfaceSeries = mSurface->seriesList();
            if (oldSurfaceSeries.size() != 1) {
                errorMsg = "No single series originally";
                break;
            }
            const auto& oldSeries = oldSurfaceSeries[0];
            const auto& oldDataProxy = oldSeries->dataProxy();
            const auto& oldDataArray = oldDataProxy->array();
            
            QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                          mBenchIdxs, mPlotParams.xIdx, "X");
            // Check subsets symmetry/min size
            bool symOK = true, minOK = true;
            QString culpritName;
            Q_ASSERT(!newBchSubsets.empty());
            int refSize = newBchSubsets.empty() ? 0 : newBchSubsets[0].idxs.size();
            for (int i = 0; symOK && minOK && i < newBchSubsets.size(); ++i) {
                symOK = newBchSubsets[i].idxs.size() == refSize;
                minOK = newBchSubsets[i].idxs.size() >= 2;
                if (!symOK || !minOK)
                    culpritName = newBchSubsets[i].name;
            }
            if (!symOK) {
                errorMsg = "Inconsistent number of X-values between benchmarks to trace surface for: " + culpritName;
                break;
            }
            else if (!minOK) {
                errorMsg = "Not enough X-values to trace surface for: " + culpritName;
                break;
            }
            // Check rows
            if (newBchSubsets.size() != oldDataProxy->rowCount()) {
                errorMsg = "Number of single series rows is different";
                break;
            }
            
            int prevRowSize = 0;
            int newRowsIdx  = 0;
            for (const auto& bchSubset : qAsConst(newBchSubsets))
            {
                // Check inter benchmark consistency
                if (prevRowSize > 0 && prevRowSize != bchSubset.idxs.size()) {
                    errorMsg = "Inconsistent number of X-values between benchmarks to trace surface";
                    break;
                }
                prevRowSize = bchSubset.idxs.size();
                
                const auto& oldRow  = oldDataArray->at(newRowsIdx);
                if (bchSubset.idxs.size() != oldRow->size())
                {
                    errorMsg = "Number of series columns is different";
                    break;
                }
                ++newRowsIdx;
            }
            
            // Direct update if compatible
            if ( errorMsg.isEmpty() )
            {
                bool custXAxis = true;
                QString custXName;
                double zFallback = 0.;
                
                newRowsIdx  = 0;
                for (const auto& bchSubset : qAsConst(newBchSubsets))
                {
                    double xFallback = 0.;
                    int newColsIdx = 0;
                    for (int idx : bchSubset.idxs)
                    {
                        // Update item
                        QString xName = newBchResults.getParamName(mPlotParams.xType == PlotArgumentType,
                                                                   idx, mPlotParams.xIdx);
                        double xVal = BenchResults::getParamValue(xName, custXName, custXAxis, xFallback);
                        double yVal = getYPlotValue(newBchResults.benchmarks[idx], mPlotParams.yType) * mCurrentTimeFactor;
                        
                        oldDataProxy->setItem(newRowsIdx, newColsIdx,
                                              QSurfaceDataItem( QVector3D(xVal, yVal, zFallback) ));
                        ++newColsIdx;
                    }
                    ++zFallback;
                    ++newRowsIdx;
                }
            }
        }
        else
        {
            // Check compatibility with previous
            const auto& oldSurfaceSeries = mSurface->seriesList();
            if (oldSurfaceSeries.empty()) {
                errorMsg = "No series originally";
                break;
            }
            const auto newBchNames = newBchResults.segment2DNames(mBenchIdxs,
                                                                  mPlotParams.xType == PlotArgumentType, mPlotParams.xIdx,
                                                                  mPlotParams.zType == PlotArgumentType, mPlotParams.zIdx);
            if (newBchNames.size() < oldSurfaceSeries.size()) {
                errorMsg = "Number of series is different";
                break;
            }
            
            int newSeriesIdx = 0;
            for (const auto& bchName : newBchNames)
            {
                QVector<BenchSubset> newBchZSubs = newBchResults.segmentParam(mPlotParams.zType == PlotArgumentType,
                                                                              bchName.idxs, mPlotParams.zIdx);
                // Ignore incompatible series
                if ( newBchZSubs.isEmpty() ) {
                    qWarning() << "No Z-value to trace surface for other benchmarks";
                    continue;
                }
                
                // Check subsets symmetry/min size
                bool symOK = true, minOK = true;
                QString culpritName;
                int refSize = newBchZSubs[0].idxs.size();
                for (int i=0; symOK && minOK && i<newBchZSubs.size(); ++i) {
                    symOK = newBchZSubs[i].idxs.size() == refSize;
                    minOK = newBchZSubs[i].idxs.size() >= 2;
                    if (!symOK || !minOK)
                        culpritName = newBchZSubs[i].name;
                }
                if (!symOK) {
                    qWarning() << "Inconsistent number of X-values between benchmarks to trace surface for:"
                               << bchName.name + "[Z=" + culpritName + "]";
                    continue;
                }
                else if (!minOK) {
                    qWarning() << "Not enough X-values to trace surface for:"
                               << bchName.name + "[Z=" + culpritName + "]";
                    continue;
                }
                
                const auto& oldSeries = oldSurfaceSeries.at(newSeriesIdx);
                const auto& oldDataProxy = oldSeries->dataProxy();
                const auto& oldDataArray = oldDataProxy->array();
                if (bchName.name != mSeriesMapping[newSeriesIdx].oldName)
                {
                    errorMsg = "Series has different name";
                    break;
                }
                if (newBchZSubs.size() != oldDataProxy->rowCount()) {
                    errorMsg = "Number of single series rows is different";
                    break;
                }
                
                int newRowsIdx  = 0;
                for (const auto& bchZSub : qAsConst(newBchZSubs))
                {
                    const auto& oldRow  = oldDataArray->at(newRowsIdx);
                    QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                                  bchZSub.idxs, mPlotParams.xIdx, "X");
                    Q_ASSERT(newBchSubsets.size() <= 1);
                    if (newBchSubsets.empty() || newBchSubsets[0].idxs.size() != oldRow->size())
                    {
                        errorMsg = "Number of series columns is different";
                        break;
                    }
                    ++newRowsIdx;
                }
                ++newSeriesIdx;
            }
            
            // Direct update if compatible
            if ( errorMsg.isEmpty() )
            {
                bool custXAxis = true, custZAxis = true;
                QString custXName, custZName;
                
                newSeriesIdx = 0;
                for (const auto& bchName : newBchNames)
                {
                    QVector<BenchSubset> newBchZSubs = newBchResults.segmentParam(mPlotParams.zType == PlotArgumentType,
                                                                                  bchName.idxs, mPlotParams.zIdx);
                    // Ignore incompatible series
                    if ( newBchZSubs.isEmpty() )
                        continue;
                    
                    // Check subsets symmetry/min size
                    bool symOK = true, minOK = true;
                    QString culpritName;
                    int refSize = newBchZSubs[0].idxs.size();
                    for (int i=0; symOK && minOK && i<newBchZSubs.size(); ++i) {
                        symOK = newBchZSubs[i].idxs.size() == refSize;
                        minOK = newBchZSubs[i].idxs.size() >= 2;
                        if (!symOK || !minOK)
                            culpritName = newBchZSubs[i].name;
                    }
                    if (!symOK || !minOK)
                        continue;
                    
                    const auto& oldSeries = oldSurfaceSeries.at(newSeriesIdx);
                    const auto& oldDataProxy = oldSeries->dataProxy();
                    
                    double zFallback = 0.;
                    int newRowsIdx  = 0;
                    for (const auto& bchZSub : qAsConst(newBchZSubs))
                    {
                        const QString zName = bchZSub.name;
                        double zVal = BenchResults::getParamValue(zName, custZName, custZAxis, zFallback);
                        
                        QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                                      bchZSub.idxs, mPlotParams.xIdx, "X");
                        Q_ASSERT(newBchSubsets.size() == 1);
                        const auto& bchSubset = newBchSubsets[0];
                        
                        double xFallback = 0.;
                        int newColsIdx = 0;
                        for (int idx : bchSubset.idxs)
                        {
                            // Update item
                            QString xName = newBchResults.getParamName(mPlotParams.xType == PlotArgumentType,
                                                                       idx, mPlotParams.xIdx);
                            double xVal = BenchResults::getParamValue(xName, custXName, custXAxis, xFallback);
                            double yVal = getYPlotValue(newBchResults.benchmarks[idx], mPlotParams.yType) * mCurrentTimeFactor;
                            
                            oldDataProxy->setItem(newRowsIdx, newColsIdx,
                                                  QSurfaceDataItem( QVector3D(xVal, yVal, zVal) ));
                            ++newColsIdx;
                        }
                        ++newRowsIdx;
                    }
                    ++newSeriesIdx;
                }
            }
        }
        
        break;  // once
    }
    
    if ( !errorMsg.isEmpty() )
    {
        // Reset update if all benchmarks
        if (mAllIndexes)
        {
            saveConfig();
            setupChart(newBchResults, mBenchIdxs, mPlotParams, false);
            setupOptions(false);
        }
        else
        {
            QMessageBox::critical(this, "Chart reload", errorMsg);
            return;
        }
    }
    
    // Restore Y-range
    QValue3DAxis* axisY = mSurface->axisY();
    if (axisY)
    {
        axisY->setMin(mAxesParams[1].min);
        axisY->setMax(mAxesParams[1].max);
    }
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() +")");
}

void Plotter3DSurface::onSnapshotClicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save snapshot"), "", tr("Images (*.png)"));
    
    if ( !fileName.isEmpty() )
    {
        QImage image = mSurface->renderToImage(8);

        bool ok = image.save(fileName, "PNG");
        if (!ok)
            QMessageBox::warning(this, "Chart snapshot", "Error saving snapshot file.");
    }
}
