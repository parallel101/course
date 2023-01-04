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

#include "plotter_3dbars.h"
#include "ui_plotter_3dbars.h"

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

static const char* config_file = "config_3dbars.json";
static const bool force_config = false;


Plotter3DBars::Plotter3DBars(const BenchResults &bchResults, const QVector<int> &bchIdxs,
                             const PlotParams &plotParams, const QString &origFilename,
                             const QVector<FileReload>& addFilenames, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Plotter3DBars)
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
    this->setWindowTitle("3D Bars - " + fileInfo.fileName());
    
    connectUI();
    
    // Init
    setupChart(bchResults, bchIdxs, plotParams);
    setupOptions();
    
    // Show
    QWidget *container = QWidget::createWindowContainer(mBars);
    ui->horizontalLayout->insertWidget(0, container, 1);
}

Plotter3DBars::~Plotter3DBars()
{
    // Save options to file
    saveConfig();
    
    delete ui;
}

void Plotter3DBars::connectUI()
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
    connect(ui->comboBoxTheme, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DBars::onComboThemeChanged);
    
    // Bars
    setupGradients();
    connect(ui->comboBoxGradient, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DBars::onComboGradientChanged);
    
    connect(ui->doubleSpinBoxThickness, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DBars::onSpinThicknessChanged);
    connect(ui->doubleSpinBoxFloor,     QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DBars::onSpinFloorChanged);
    connect(ui->doubleSpinBoxSpacingX,  QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DBars::onSpinSpaceXChanged);
    connect(ui->doubleSpinBoxSpacingZ,  QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DBars::onSpinSpaceZChanged);
    connect(ui->pushButtonSeries,       &QPushButton::clicked, this, &Plotter3DBars::onSeriesEditClicked);
    
    if (!isYTimeBased(mPlotParams.yType))
        ui->comboBoxTimeUnit->setEnabled(false);
    else
    {
        ui->comboBoxTimeUnit->addItem("ns", 1000.);
        ui->comboBoxTimeUnit->addItem("us", 1.);
        ui->comboBoxTimeUnit->addItem("ms", 0.001);
        connect(ui->comboBoxTimeUnit, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DBars::onComboTimeUnitChanged);
    }
    
    // Axes
    ui->comboBoxAxis->addItem("X-Axis");
    ui->comboBoxAxis->addItem("Y-Axis");
    ui->comboBoxAxis->addItem("Z-Axis");
    connect(ui->comboBoxAxis, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DBars::onComboAxisChanged);
    
    connect(ui->checkBoxAxisRotate,  &QCheckBox::stateChanged, this, &Plotter3DBars::onCheckAxisRotate);
    connect(ui->checkBoxTitle,       &QCheckBox::stateChanged, this, &Plotter3DBars::onCheckTitleVisible);
    connect(ui->checkBoxLog,         &QCheckBox::stateChanged, this, &Plotter3DBars::onCheckLog);
    connect(ui->spinBoxLogBase,      QOverload<int>::of(&QSpinBox::valueChanged), this, &Plotter3DBars::onSpinLogBaseChanged);
    connect(ui->lineEditTitle,       &QLineEdit::textChanged, this, &Plotter3DBars::onEditTitleChanged);
    connect(ui->lineEditFormat,      &QLineEdit::textChanged, this, &Plotter3DBars::onEditFormatChanged);
    connect(ui->doubleSpinBoxMin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DBars::onSpinMinChanged);
    connect(ui->doubleSpinBoxMax,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &Plotter3DBars::onSpinMaxChanged);
    connect(ui->comboBoxMin,         QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DBars::onComboMinChanged);
    connect(ui->comboBoxMax,         QOverload<int>::of(&QComboBox::currentIndexChanged), this, &Plotter3DBars::onComboMaxChanged);
    connect(ui->spinBoxTicks,        QOverload<int>::of(&QSpinBox::valueChanged), this, &Plotter3DBars::onSpinTicksChanged);
    connect(ui->spinBoxMTicks,       QOverload<int>::of(&QSpinBox::valueChanged), this, &Plotter3DBars::onSpinMTicksChanged);
    
    // Actions
    connect(&mWatcher,              &QFileSystemWatcher::fileChanged, this, &Plotter3DBars::onAutoReload);
    connect(ui->checkBoxAutoReload, &QCheckBox::stateChanged, this, &Plotter3DBars::onCheckAutoReload);
    connect(ui->pushButtonReload,   &QPushButton::clicked, this, &Plotter3DBars::onReloadClicked);
    connect(ui->pushButtonSnapshot, &QPushButton::clicked, this, &Plotter3DBars::onSnapshotClicked);
}

void Plotter3DBars::setupChart(const BenchResults &bchResults, const QVector<int> &bchIdxs, const PlotParams &plotParams, bool init)
{
    QScopedPointer<Q3DBars> scopedBars;
    Q3DBars* bars = nullptr;
    if (init) {
        scopedBars.reset( new Q3DBars() );
        bars = scopedBars.get();
    }
    else {  // Re-init
        bars = mBars;
        const auto seriesList = bars->seriesList();
        for (const auto barSeries : seriesList)
            bars->removeSeries(barSeries);
        const auto barsAxes = bars->axes();
        for (const auto axis : barsAxes)
            bars->releaseAxis(axis);
        mSeriesMapping.clear();
    }
    Q_ASSERT(bars);
    
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
    bool hasZParam = plotParams.zType != PlotEmptyType;
    
    //
    // No Z-param -> one row per benchmark type
    if (!hasZParam)
    {
        // Single series (i.e. color)
        QScopedPointer<QBar3DSeries> series(new QBar3DSeries);

        QVector<BenchSubset> bchSubsets = bchResults.groupParam(plotParams.xType == PlotArgumentType,
                                                                bchIdxs, plotParams.xIdx, "X");
        bool firstCol = true;
        for (const auto& bchSubset : qAsConst(bchSubsets))
        {
            // One row per benchmark * X-group
            QScopedPointer<QBarDataRow> data(new QBarDataRow);
            
            const QString & subsetName = bchSubset.name;
//            qDebug() << "subsetName:" << subsetName;
//            qDebug() << "subsetIdxs:" << bchSubset.idxs;
            
            QStringList colLabels;
            for (int idx : bchSubset.idxs)
            {
                QString xName = bchResults.getParamName(plotParams.xType == PlotArgumentType,
                                                        idx, plotParams.xIdx);
                colLabels.append(xName);
                
                // Add column
                data->append( static_cast<float>(getYPlotValue(bchResults.benchmarks[idx], plotParams.yType) * mCurrentTimeFactor) );
            }
            // Add benchmark row
            series->dataProxy()->addRow(data.take(), subsetName);
            
            // Set column labels (only if no collision, empty otherwise)
            if (firstCol) // init
                series->dataProxy()->setColumnLabels(colLabels);
            else if ( commonPartEqual(series->dataProxy()->columnLabels(), colLabels) ) {
                if (series->dataProxy()->columnLabels().size() < colLabels.size()) // replace by longest
                    series->dataProxy()->setColumnLabels(colLabels);
            }
            else { // collision
                series->dataProxy()->setColumnLabels( QStringList("") );
            }
            firstCol = false;
//            qDebug() << "[Multi-NoZ] colLabels:" << colLabels << "|" << series->dataProxy()->columnLabels();
        }
        // Add series
        series->setItemLabelFormat(QStringLiteral("@rowLabel [X=@colLabel]: @valueLabel"));
        series->setMesh(QAbstract3DSeries::MeshBevelBar);
        series->setMeshSmooth(false);
        mSeriesMapping.push_back({"", ""}); // color set later
        
        bars->addSeries(series.take());
    }
    //
    // Z-param -> one series per benchmark, one row per Z, one column per X
    else
    {
        // Initial segmentation by 'full name % param1 % param2' (group benchmarks)
        const auto bchNames = bchResults.segment2DNames(bchIdxs,
                                                        plotParams.xType == PlotArgumentType, plotParams.xIdx,
                                                        plotParams.zType == PlotArgumentType, plotParams.zIdx);
        QStringList prevRowLabels, prevColLabels;
        bool sameRowLabels = true, sameColLabels = true;
        for (const auto& bchName : bchNames)
        {
            // One series (i.e. color) per 2D name
            QScopedPointer<QBar3DSeries> series(new QBar3DSeries);
//            qDebug() << "bchName" << bchName.name << "|" << bchName.idxs;
            
            // Segment: one sub per Z-param from 2D names
            QVector<BenchSubset> bchZSubs = bchResults.segmentParam(plotParams.zType == PlotArgumentType,
                                                                    bchName.idxs, plotParams.zIdx);
            QStringList curRowLabels;
            for (const auto& bchZSub : qAsConst(bchZSubs))
            {
//                qDebug() << "bchZSub" << bchZSub.name << "|" << bchZSub.idxs;
                curRowLabels.append(bchZSub.name);
                
                // One row per Z-param from 2D names
                QScopedPointer<QBarDataRow> data(new QBarDataRow);
                
                // Group: one column per X-param
                QVector<BenchSubset> bchSubsets = bchResults.groupParam(plotParams.xType == PlotArgumentType,
                                                                        bchZSub.idxs, plotParams.xIdx, "X");
                Q_ASSERT(bchSubsets.size() == 1);
                if (bchSubsets.empty()) {
                    qWarning() << "Missing X-parameter subset for Z-row: " << bchZSub.name;
                    break;
                }
                const auto& bchSubset = bchSubsets[0];
                
                QStringList curColLabels;
                for (int idx : bchSubset.idxs)
                {
                    QString xName = bchResults.getParamName(plotParams.xType == PlotArgumentType,
                                                            idx, plotParams.xIdx);
                    curColLabels.append(xName);
                    
                    // Y-values on row
                    data->append( static_cast<float>(getYPlotValue(bchResults.benchmarks[idx], plotParams.yType) * mCurrentTimeFactor) );
                }
                // Add benchmark row
                series->dataProxy()->addRow(data.take());
                                    
                // Check column labels collisions
                if (sameColLabels) {
                    if ( prevColLabels.isEmpty() ) // init
                        prevColLabels = curColLabels;
                    else {
                        if ( commonPartEqual(prevColLabels, curColLabels) ) {
                            if (prevColLabels.size() < curColLabels.size()) // replace by longest
                                prevColLabels = curColLabels;
                        }
                        else sameColLabels = false;
                    }
                }
//                qDebug() << "[Multi-Z] curColLabels:" << curColLabels << "|" << prevColLabels;
            }
            // Check row labels collisions
            if (sameRowLabels) {
                if ( prevRowLabels.isEmpty() ) // init
                    prevRowLabels = curRowLabels;
                else {
                    if ( commonPartEqual(prevRowLabels, curRowLabels) ) {
                        if (prevRowLabels.size() < curRowLabels.size()) // replace by longest
                            prevRowLabels = curRowLabels;
                    }
                    else sameRowLabels = false;
                }
            }
//            qDebug() << "[Multi-Z] curRowLabels:" << curRowLabels << "|" << prevRowLabels;
            //
            // Add series
            series->setName( bchName.name );
            mSeriesMapping.push_back({bchName.name, bchName.name}); // color set later
            series->setItemLabelFormat(QStringLiteral("@seriesName [@colLabel, @rowLabel]: @valueLabel"));
            series->setMesh(QAbstract3DSeries::MeshBevelBar);
            series->setMeshSmooth(false);
            
            bars->addSeries(series.take());
        }
        // Set row/column labels (empty if collisions)
        if ( !bars->seriesList().isEmpty() && bars->seriesList().constFirst()->dataProxy()->rowCount() > 0)
        {
            for (auto &series : bars->seriesList()) {
                series->dataProxy()->setColumnLabels(sameColLabels ? prevColLabels : QStringList(""));
                series->dataProxy()->setRowLabels(   sameRowLabels ? prevRowLabels : QStringList(""));
            }
        }
    }
    
    // Axes
    if ( !bars->seriesList().isEmpty() && bars->seriesList().constFirst()->dataProxy()->rowCount() > 0)
    {
        // General
        bars->setShadowQuality(QAbstract3DGraph::ShadowQualitySoftMedium);
        
        // X-axis
        QCategory3DAxis *colAxis = bars->columnAxis();
        if (plotParams.xType == PlotArgumentType)
            colAxis->setTitle("Argument " + QString::number(plotParams.xIdx+1));
        else if (plotParams.xType == PlotTemplateType)
            colAxis->setTitle("Template " + QString::number(plotParams.xIdx+1));
        if (plotParams.xType != PlotEmptyType)
            colAxis->setTitleVisible(true);
        
        // Y-axis
        QValue3DAxis *valAxis = bars->valueAxis();
        valAxis->setTitle( getYPlotName(plotParams.yType, bchResults.meta.time_unit) );
        valAxis->setTitleVisible(true);
        
        // Z-axis
        if (plotParams.zType != PlotEmptyType)
        {
            QCategory3DAxis *rowAxis = bars->rowAxis();
            if (plotParams.zType == PlotArgumentType)
                rowAxis->setTitle("Argument " + QString::number(plotParams.zIdx+1));
            else
                rowAxis->setTitle("Template " + QString::number(plotParams.zIdx+1));
            rowAxis->setTitleVisible(true);
        }
    }
    else {
        // Title-like
        QCategory3DAxis *colAxis = bars->columnAxis();
        colAxis->setTitle("No compatible series to display");
        colAxis->setTitleVisible(true);
    }
    
    if (init)
    {
        // Take
        mBars = scopedBars.take();
    }
}

void Plotter3DBars::setupOptions(bool init)
{
    // General
    if (init) {
        mBars->activeTheme()->setType(Q3DTheme::ThemePrimaryColors);
    }
    
    mIgnoreEvents = true;
    int prevAxisIdx = ui->comboBoxAxis->currentIndex();
    
    if (!init)  // Re-init
    {
        ui->comboBoxAxis->setCurrentIndex(0);
        for (auto &axisParams : mAxesParams)
            axisParams.reset();
        ui->comboBoxMin->clear();
        ui->comboBoxMax->clear();
        ui->checkBoxAxisRotate->setChecked(false);
        ui->checkBoxTitle->setChecked(true);
        ui->checkBoxLog->setChecked(false);
        ui->comboBoxGradient->setCurrentIndex(0);
    }
    
    // Time unit
    if      (mCurrentTimeFactor > 1.) ui->comboBoxTimeUnit->setCurrentIndex(0); // ns
    else if (mCurrentTimeFactor < 1.) ui->comboBoxTimeUnit->setCurrentIndex(2); // ms
    else                              ui->comboBoxTimeUnit->setCurrentIndex(1); // us
    
    // Axes
    // X-axis
    QCategory3DAxis *colAxis = mBars->columnAxis();
    if (colAxis)
    {
        auto& axisParams = mAxesParams[0];
        
        axisParams.titleText = colAxis->title();
        axisParams.title = !axisParams.titleText.isEmpty();
        
        ui->doubleSpinBoxMin->setVisible(false);
        ui->doubleSpinBoxMax->setVisible(false);
        
        ui->lineEditTitle->setText( axisParams.titleText );
        ui->lineEditTitle->setCursorPosition(0);
        if ( !colAxis->labels().isEmpty() && !colAxis->labels().constFirst().isEmpty() ) {
            axisParams.range = colAxis->labels();
        }
        else if ( !mBars->seriesList().isEmpty() ) {
            int maxCol = 0;
            const auto& seriesList = mBars->seriesList();
            for (const auto& series : seriesList)
                for (int iR=0; iR < series->dataProxy()->rowCount(); ++iR)
                    if (maxCol < series->dataProxy()->rowAt(iR)->size())
                        maxCol = series->dataProxy()->rowAt(iR)->size();
            for (int i=0; i<maxCol; ++i)
                axisParams.range.append( QString::number(i+1) );
        }
        ui->comboBoxMin->addItems( axisParams.range );
        ui->comboBoxMax->addItems( axisParams.range );
        ui->comboBoxMax->setCurrentIndex(ui->comboBoxMax->count() - 1);
        axisParams.maxIdx = ui->comboBoxMax->count() - 1;
    }
    // Y-axis
    QValue3DAxis *valAxis = mBars->valueAxis();
    if (valAxis)
    {
        auto& axisParams = mAxesParams[1];
        
        axisParams.titleText = valAxis->title();
        axisParams.title = !axisParams.titleText.isEmpty();
        
        ui->doubleSpinBoxFloor->setMinimum( valAxis->min() );
        ui->doubleSpinBoxFloor->setMaximum( valAxis->max() );
        ui->lineEditFormat->setText( valAxis->labelFormat() );
        ui->lineEditFormat->setCursorPosition(0);
        ui->doubleSpinBoxMin->setValue( valAxis->min() );
        ui->doubleSpinBoxMax->setValue( valAxis->max() );
        ui->spinBoxTicks->setValue( valAxis->segmentCount() );
        ui->spinBoxMTicks->setValue( valAxis->subSegmentCount() );
    }
    // Z-axis
    QCategory3DAxis *rowAxis = mBars->rowAxis();
    if (rowAxis)
    {
        auto& axisParams = mAxesParams[2];
        
        axisParams.titleText = rowAxis->title();
        axisParams.title = !axisParams.titleText.isEmpty();
        if ( !rowAxis->labels().isEmpty() && !rowAxis->labels().constFirst().isEmpty() ) {
            axisParams.range = rowAxis->labels();
        }
        else if ( !mBars->seriesList().isEmpty() ) {
            int maxRow = 0;
            const auto& seriesList = mBars->seriesList();
            for (const auto& series : seriesList)
                if (maxRow < series->dataProxy()->rowCount())
                    maxRow = series->dataProxy()->rowCount();
            for (int i=0; i<maxRow; ++i)
                axisParams.range.append( QString::number(i+1) );
        }
        axisParams.maxIdx = axisParams.range.size()-1;
    }
    mIgnoreEvents = false;
    
    
    // Load options from file
    loadConfig(init);
    
    
    // Apply actions
    if (ui->checkBoxAutoReload->isChecked())
        onCheckAutoReload(Qt::Checked);
    
    // Update series color config
    const auto& chartSeries = mBars->seriesList();
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
    ui->labelLastReload->setText("(Last: " + now.toString() +")");
}

void Plotter3DBars::loadConfig(bool init)
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
        
        // Bars
        if (json.contains("bars.gradient") && json["bars.gradient"].isString())
            ui->comboBoxGradient->setCurrentText( json["bars.gradient"].toString() );
        if (json.contains("bars.thick") && json["bars.thick"].isDouble())
            ui->doubleSpinBoxThickness->setValue( json["bars.thick"].toDouble() );
        if (json.contains("bars.floor") && json["bars.floor"].isDouble())
            ui->doubleSpinBoxFloor->setValue( json["bars.floor"].toDouble() );
        if (json.contains("bars.spacing.x") && json["bars.spacing.x"].isDouble())
            ui->doubleSpinBoxSpacingX->setValue( json["bars.spacing.x"].toDouble() );
        if (json.contains("bars.spacing.z") && json["bars.spacing.z"].isDouble())
            ui->doubleSpinBoxSpacingZ->setValue( json["bars.spacing.z"].toDouble() );
        
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
            if (!init)
            {
                if (json.contains(prefix + ".titleText") && json[prefix + ".titleText"].isString()) {
                    axis.titleText = json[prefix + ".titleText"].toString();
                    ui->lineEditTitle->setText( axis.titleText );
                    ui->lineEditTitle->setCursorPosition(0);
                }
            }
            // x or z-axis
            if (idx == 0 || idx == 2)
            {
                if (force_config)
                {
                    if (json.contains(prefix + ".min") && json[prefix + ".min"].isString()) {
                        ui->comboBoxMin->setCurrentText( json[prefix + ".min"].toString() );
                        axis.minIdx = ui->comboBoxMin->currentIndex();
                    }
                    if (json.contains(prefix + ".max") && json[prefix + ".max"].isString()) {
                        ui->comboBoxMax->setCurrentText( json[prefix + ".max"].toString() );
                        axis.maxIdx = ui->comboBoxMax->currentIndex();
                    }
                }
                if (idx == 0)
                {
                    prefix = "axis.y";
                    ui->comboBoxAxis->setCurrentIndex(1);
                }
            }
            else // y-axis
            {
                if (json.contains(prefix + ".log") && json[prefix + ".log"].isBool())
                    ui->checkBoxLog->setChecked( json[prefix + ".log"].toBool() );
                if (json.contains(prefix + ".logBase") && json[prefix + ".logBase"].isDouble())
                    ui->spinBoxLogBase->setValue( json[prefix + ".logBase"].toInt(10) );
                if (json.contains(prefix + ".labelFormat") && json[prefix + ".labelFormat"].isString()) {
                    ui->lineEditFormat->setText( json[prefix + ".labelFormat"].toString() );
                    ui->lineEditFormat->setCursorPosition(0);
                }
                if (json.contains(prefix + ".ticks") && json[prefix + ".ticks"].isDouble())
                    ui->spinBoxTicks->setValue( json[prefix + ".ticks"].toInt(5) );
                if (json.contains(prefix + ".mticks") && json[prefix + ".mticks"].isDouble())
                    ui->spinBoxMTicks->setValue( json[prefix + ".mticks"].toInt(1) );
                if (!init)
                {
                    if (json.contains(prefix + ".min") && json[prefix + ".min"].isDouble())
                        ui->doubleSpinBoxMin->setValue( json[prefix + ".min"].toDouble() );
                    if (json.contains(prefix + ".max") && json[prefix + ".max"].isDouble())
                        ui->doubleSpinBoxMax->setValue( json[prefix + ".max"].toDouble() );
                }
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

void Plotter3DBars::saveConfig()
{
    QFile configFile(QString(config_folder) + config_file);
    if (configFile.open(QIODevice::WriteOnly))
    {
        QJsonObject json;
        
        // Theme
        json["theme"] = ui->comboBoxTheme->currentText();
        // Bars
        json["bars.gradient"]  = ui->comboBoxGradient->currentText();
        json["bars.thick"]     = ui->doubleSpinBoxThickness->value();
        json["bars.floor"]     = ui->doubleSpinBoxFloor->value();
        json["bars.spacing.x"] = ui->doubleSpinBoxSpacingX->value();
        json["bars.spacing.z"] = ui->doubleSpinBoxSpacingZ->value();
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
            
            json[prefix + ".rotate"]    = axis.rotate;
            json[prefix + ".title"]     = axis.title;
            json[prefix + ".titleText"] = axis.titleText;
            // x or z-axis
            if (idx == 0 || idx == 2)
            {
                if ( axis.minIdx >= 0 && axis.minIdx < axis.range.size()
                  && axis.maxIdx >= 0 && axis.maxIdx < axis.range.size() ) {
                    json[prefix + ".min"] = axis.range[axis.minIdx];
                    json[prefix + ".max"] = axis.range[axis.maxIdx];
                }
                prefix = "axis.y";
            }
            else // y-axis
            {
                json[prefix + ".log"]         = ui->checkBoxLog->isChecked();
                json[prefix + ".logBase"]     = ui->spinBoxLogBase->value();
                json[prefix + ".labelFormat"] = ui->lineEditFormat->text();
                json[prefix + ".min"]         = ui->doubleSpinBoxMin->value();
                json[prefix + ".max"]         = ui->doubleSpinBoxMax->value();
                json[prefix + ".ticks"]       = ui->spinBoxTicks->value();
                json[prefix + ".mticks"]      = ui->spinBoxMTicks->value();
                
                prefix = "axis.z";
            }
        }
        
        configFile.write( QJsonDocument(json).toJson() );
    }
    else
        qWarning() << "Couldn't update: " << QString(config_folder) + config_file;
}

void Plotter3DBars::setupGradients()
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
void Plotter3DBars::onComboThemeChanged(int index)
{
    Q3DTheme::Theme theme = static_cast<Q3DTheme::Theme>(
                ui->comboBoxTheme->itemData(index).toInt());
    mBars->activeTheme()->setType(theme);
    
    onComboGradientChanged( ui->comboBoxGradient->currentIndex() );
    
    // Update series color
    const auto& chartSeries = mBars->seriesList();
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
// Bars
void Plotter3DBars::onComboGradientChanged(int idx)
{
    if (idx == 0)
    {
        for (auto& series : mBars->seriesList())
            series->setColorStyle(Q3DTheme::ColorStyleUniform);
    }
    else
    {
        for (auto& series : mBars->seriesList()) {
            series->setBaseGradient( mGrads[idx-1] );
            series->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
        }
    }
}

void Plotter3DBars::onSpinThicknessChanged(double d)
{
    mBars->setBarThickness(d);
}

void Plotter3DBars::onSpinFloorChanged(double d)
{
    mBars->setFloorLevel(d);
}

void Plotter3DBars::onSpinSpaceXChanged(double d)
{
    QSizeF barSpacing = mBars->barSpacing();
    barSpacing.setWidth(d);
    
    mBars->setBarSpacing(barSpacing);
}

void Plotter3DBars::onSpinSpaceZChanged(double d)
{
    QSizeF barSpacing = mBars->barSpacing();
    barSpacing.setHeight(d);
    
    mBars->setBarSpacing(barSpacing);
}

void Plotter3DBars::onSeriesEditClicked()
{
    SeriesDialog seriesDialog(mSeriesMapping, this);
    auto res = seriesDialog.exec();
    if (res == QDialog::Accepted)
    {
        const auto& chartSeries = mBars->seriesList();
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

void Plotter3DBars::onComboTimeUnitChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    
    // Update data
    double unitFactor = ui->comboBoxTimeUnit->currentData().toDouble();
    double updateFactor = unitFactor / mCurrentTimeFactor;  // can cause precision loss
    auto chartSeries = mBars->seriesList();
    if (chartSeries.empty())
        return;
    
    for (auto& series : chartSeries)
    {
        const auto& dataProxy = series->dataProxy();
        for (int iR = 0; iR < dataProxy->rowCount(); ++iR)
        {
            auto row = dataProxy->rowAt(iR);
            for (int iC = 0; iC < row->size(); ++iC)
            {
                auto item = dataProxy->itemAt(iR, iC);
                dataProxy->setItem(iR, iC,
                                   QBarDataItem( static_cast<float>(item->value() * updateFactor) ));
            }
        }
    }
    
    // Update axis title
    QString oldUnitName = "(us)";
    if      (mCurrentTimeFactor > 1.) oldUnitName = "(ns)";
    else if (mCurrentTimeFactor < 1.) oldUnitName = "(ms)";
    
    auto yAxis = mBars->valueAxis();
    if (yAxis) {
        QString axisTitle = yAxis->title();
        if (axisTitle.endsWith(oldUnitName)) {
            QString unitName  = ui->comboBoxTimeUnit->currentText();
            onEditTitleChanged2(axisTitle.replace(axisTitle.size() - 3, 2, unitName), 1);
        }
    }
    // Update range
    if (updateFactor > 1.) {    // enforce proper order
        ui->doubleSpinBoxMax->setValue(ui->doubleSpinBoxMax->value() * updateFactor);
        ui->doubleSpinBoxMin->setValue(ui->doubleSpinBoxMin->value() * updateFactor);
    }
    else {
        ui->doubleSpinBoxMin->setValue(ui->doubleSpinBoxMin->value() * updateFactor);
        ui->doubleSpinBoxMax->setValue(ui->doubleSpinBoxMax->value() * updateFactor);
    }
    
    mCurrentTimeFactor = unitFactor;
}

//
// Axes
void Plotter3DBars::onComboAxisChanged(int idx)
{
    // Update UI
    bool wasIgnoring = mIgnoreEvents;
    mIgnoreEvents = true;
    
    ui->checkBoxAxisRotate->setChecked( mAxesParams[idx].rotate );
    ui->checkBoxTitle->setChecked( mAxesParams[idx].title );
    ui->checkBoxLog->setEnabled( idx == 1 );
    ui->spinBoxLogBase->setEnabled( ui->checkBoxLog->isEnabled() && ui->checkBoxLog->isChecked() );
    ui->lineEditTitle->setText( mAxesParams[idx].titleText );
    ui->lineEditTitle->setCursorPosition(0);
    ui->lineEditFormat->setEnabled( idx == 1 );
    // Force visibility order
    if (idx == 1) {
        ui->comboBoxMin->setVisible(false);
        ui->comboBoxMax->setVisible(false);
        ui->doubleSpinBoxMin->setVisible(true);
        ui->doubleSpinBoxMax->setVisible(true);
    }
    else {
        ui->doubleSpinBoxMin->setVisible(false);
        ui->doubleSpinBoxMax->setVisible(false);
        ui->comboBoxMin->setVisible(true);
        ui->comboBoxMax->setVisible(true);
        
        ui->comboBoxMin->clear();
        ui->comboBoxMax->clear();
        ui->comboBoxMin->addItems( mAxesParams[idx].range );
        ui->comboBoxMax->addItems( mAxesParams[idx].range );
        ui->comboBoxMin->setCurrentIndex( mAxesParams[idx].minIdx );
        ui->comboBoxMax->setCurrentIndex( mAxesParams[idx].maxIdx );
    }
    ui->spinBoxTicks->setEnabled(  idx == 1 && !ui->checkBoxLog->isChecked() );
    ui->spinBoxMTicks->setEnabled( idx == 1 && !ui->checkBoxLog->isChecked() );
    
    mIgnoreEvents = wasIgnoring;
}

void Plotter3DBars::onCheckAxisRotate(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QAbstract3DAxis* axis;
    if      (iAxis == 0)    axis = mBars->columnAxis();
    else if (iAxis == 1)    axis = mBars->valueAxis();
    else                    axis = mBars->rowAxis();
    
    if (axis) {
        axis->setTitleFixed(state != Qt::Checked);
        axis->setLabelAutoRotation(state == Qt::Checked ? 90 : 0);
        mAxesParams[iAxis].rotate = state == Qt::Checked;
    }
}

void Plotter3DBars::onCheckTitleVisible(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QAbstract3DAxis* axis;
    if      (iAxis == 0)    axis = mBars->columnAxis();
    else if (iAxis == 1)    axis = mBars->valueAxis();
    else                    axis = mBars->rowAxis();
    
    if (axis) {
        axis->setTitleVisible(state == Qt::Checked);
        mAxesParams[iAxis].title = state == Qt::Checked;
    }
}

void Plotter3DBars::onCheckLog(int state)
{
    if (mIgnoreEvents) return;
    Q_ASSERT(ui->comboBoxAxis->currentIndex() == 1);
    QValue3DAxis* axis = mBars->valueAxis();
    
    if (axis)
    {
        if (state == Qt::Checked) {
            axis->setFormatter(new QLogValue3DAxisFormatter());
            ui->doubleSpinBoxMin->setMinimum(0.001);
        }
        else {
            axis->setFormatter(new QValue3DAxisFormatter());
            ui->doubleSpinBoxMin->setMinimum(0.);
        }
        ui->spinBoxTicks->setEnabled(  state != Qt::Checked);
        ui->spinBoxMTicks->setEnabled( state != Qt::Checked);
        ui->spinBoxLogBase->setEnabled(state == Qt::Checked);
    }
}

void Plotter3DBars::onSpinLogBaseChanged(int i)
{
    if (mIgnoreEvents) return;
    Q_ASSERT(ui->comboBoxAxis->currentIndex() == 1);
    QValue3DAxis* axis = mBars->valueAxis();
    
    if (axis)
    {
        QLogValue3DAxisFormatter* formatter = (QLogValue3DAxisFormatter*)axis->formatter();
        if (formatter)
            formatter->setBase(i);
    }
}

void Plotter3DBars::onEditTitleChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onEditTitleChanged2(text, iAxis);
}

void Plotter3DBars::onEditTitleChanged2(const QString& text, int iAxis)
{
    QAbstract3DAxis* axis;
    if      (iAxis == 0)    axis = mBars->columnAxis();
    else if (iAxis == 1)    axis = mBars->valueAxis();
    else                    axis = mBars->rowAxis();
    
    if (axis) {
        axis->setTitle(text);
        mAxesParams[iAxis].titleText = text;
    }
}

void Plotter3DBars::onEditFormatChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    Q_ASSERT(ui->comboBoxAxis->currentIndex() == 1);
    QValue3DAxis* axis = mBars->valueAxis();
    
    if (axis) {
        axis->setLabelFormat(text);
    }
}

void Plotter3DBars::onSpinMinChanged(double d)
{
    if (mIgnoreEvents) return;
    QAbstract3DAxis* axis = mBars->valueAxis();
    
    if (axis) {
        axis->setMin(d);
    }
}

void Plotter3DBars::onSpinMaxChanged(double d)
{
    if (mIgnoreEvents) return;
    QAbstract3DAxis* axis = mBars->valueAxis();
    
    if (axis) {
        axis->setMax(d);
    }
}

void Plotter3DBars::onComboMinChanged(int index)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QAbstract3DAxis* axis;
    if      (iAxis == 0)    axis = mBars->columnAxis();
    else                    axis = mBars->rowAxis();
    
    if (axis) {
        axis->setMin(index);
        mAxesParams[iAxis].minIdx = index;
    }
}

void Plotter3DBars::onComboMaxChanged(int index)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    QAbstract3DAxis* axis;
    if      (iAxis == 0)    axis = mBars->columnAxis();
    else                    axis = mBars->rowAxis();
    
    if (axis) {
        axis->setMax(index);
        mAxesParams[iAxis].maxIdx = index;
    }
}

void Plotter3DBars::onSpinTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    Q_ASSERT(ui->comboBoxAxis->currentIndex() == 1);
    QValue3DAxis* axis = mBars->valueAxis();
    
    if (axis) {
        axis->setSegmentCount(i);
    }
}

void Plotter3DBars::onSpinMTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    Q_ASSERT(ui->comboBoxAxis->currentIndex() == 1);
    QValue3DAxis* axis = mBars->valueAxis();
    
    if (axis) {
        axis->setSubSegmentCount(i);
    }
}

//
// Actions
void Plotter3DBars::onCheckAutoReload(int state)
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

void Plotter3DBars::onAutoReload(const QString &path)
{
    QFileInfo fi(path);
    if (fi.exists() && fi.isReadable() && fi.size() > 0)
        onReloadClicked();
    else
        qWarning() << "Unable to auto-reload file: " << path;
}

void Plotter3DBars::onReloadClicked()
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
            const auto& oldBarsSeries = mBars->seriesList();
            if (oldBarsSeries.size() != 1) {
                errorMsg = "No single series originally";
                break;
            }
            const auto oldSeries = oldBarsSeries[0];
            const auto oldDataProxy = oldSeries->dataProxy();
    
            QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                          mBenchIdxs, mPlotParams.xIdx, "X");
            if (newBchSubsets.size() != oldDataProxy->rowCount()) {
                errorMsg = "Number of single series rows is different";
                break;
            }
            
            int newRowsIdx = 0;
            for (const auto& bchSubset : qAsConst(newBchSubsets))
            {
                const auto& oldRowLabel = oldDataProxy->rowLabels().at(newRowsIdx);
                const QString& subsetName = bchSubset.name;
                if (subsetName != oldRowLabel)
                {
                    errorMsg = "Series row has different name";
                    break;
                }
                const auto& oldRow = oldDataProxy->rowAt(newRowsIdx);
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
                newRowsIdx = 0;
                for (const auto& bchSubset : qAsConst(newBchSubsets))
                {
                    int newColsIdx = 0;
                    for (int idx : bchSubset.idxs)
                    {
                        // Update item
                        oldDataProxy->setItem(newRowsIdx, newColsIdx,
                                              QBarDataItem( static_cast<float>(getYPlotValue(newBchResults.benchmarks[idx], mPlotParams.yType) * mCurrentTimeFactor) ));
                        ++newColsIdx;
                    }
                    ++newRowsIdx;
                }
            }
        }
        else
        {
            // Check compatibility with previous
            const auto& oldBarsSeries = mBars->seriesList();
            if (oldBarsSeries.empty()) {
                errorMsg = "No series originally";
                break;
            }
            
            const auto newBchNames = newBchResults.segment2DNames(mBenchIdxs,
                                                                 mPlotParams.xType == PlotArgumentType, mPlotParams.xIdx,
                                                                 mPlotParams.zType == PlotArgumentType, mPlotParams.zIdx);
            if (newBchNames.size() != oldBarsSeries.size()) {
                errorMsg = "Number of series is different";
                break;
            }
            
            int newSeriesIdx = 0;
            for (const auto& bchName : newBchNames)
            {
                const auto& oldSeries = oldBarsSeries.at(newSeriesIdx);
                const auto& oldDataProxy = oldSeries->dataProxy();
                if (bchName.name != mSeriesMapping[newSeriesIdx].oldName)
                {
                    errorMsg = "Series has different name";
                    break;
                }
                
                QVector<BenchSubset> newBchZSubs = newBchResults.segmentParam(mPlotParams.zType == PlotArgumentType,
                                                                              bchName.idxs, mPlotParams.zIdx);
                if (newBchZSubs.size() != oldDataProxy->rowCount()) {
                    errorMsg = "Number of series rows is different";
                    break;
                }
                
                int newRowsIdx = 0;
                for (const auto& bchZSub : qAsConst(newBchZSubs))
                {
                    const auto& oldRowLabel = oldDataProxy->rowLabels().size() < newRowsIdx ?
                                                oldDataProxy->rowLabels().at(newRowsIdx) : "";
                    const QString& subsetName = bchZSub.name;
                    if (subsetName != oldRowLabel && !oldRowLabel.isEmpty())
                    {
                        errorMsg = "Series row has different name";
                        break;
                    }
    
                    const auto& oldRow = oldDataProxy->rowAt(newRowsIdx);
                    QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                                  bchZSub.idxs, mPlotParams.xIdx, "X");
                    Q_ASSERT(newBchSubsets.size() == 1);
                    if (newBchSubsets.empty()) {
                        qWarning() << "Missing X-parameter subset for Z-row: " << bchZSub.name;
                        break;
                    }
                    if (newBchSubsets[0].idxs.size() != oldRow->size())
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
                newSeriesIdx = 0;
                for (const auto& bchName : newBchNames)
                {
                    const auto& oldSeries = oldBarsSeries.at(newSeriesIdx);
                    const auto& oldDataProxy = oldSeries->dataProxy();
                    QVector<BenchSubset> newBchZSubs = newBchResults.segmentParam(mPlotParams.zType == PlotArgumentType,
                                                                                  bchName.idxs, mPlotParams.zIdx);
                    int newRowsIdx = 0;
                    for (const auto& bchZSub : qAsConst(newBchZSubs))
                    {
                        QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                                      bchZSub.idxs, mPlotParams.xIdx, "X");
                        if (newBchSubsets.empty())
                            break;
                        const auto& bchSubset = newBchSubsets[0];
                        
                        int newColsIdx = 0;
                        for (int idx : bchSubset.idxs)
                        {
                            // Update item
                            oldDataProxy->setItem(newRowsIdx, newColsIdx,
                                                  QBarDataItem( static_cast<float>(getYPlotValue(newBchResults.benchmarks[idx], mPlotParams.yType) * mCurrentTimeFactor) ));
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
    onSpinMinChanged( ui->doubleSpinBoxMin->value() ); // force update
    onSpinMaxChanged( ui->doubleSpinBoxMax->value() );
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() +")");
}

void Plotter3DBars::onSnapshotClicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save snapshot"), "", tr("Images (*.png)"));
    
    if ( !fileName.isEmpty() )
    {
        QImage image = mBars->renderToImage(8);

        bool ok = image.save(fileName, "PNG");
        if (!ok)
            QMessageBox::warning(this, "Chart snapshot", "Error saving snapshot file.");
    }
}
