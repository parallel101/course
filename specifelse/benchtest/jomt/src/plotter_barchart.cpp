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

#include "plotter_barchart.h"
#include "ui_plotter_barchart.h"

#include "benchmark_results.h"
#include "result_parser.h"

#include <QFileInfo>
#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonObject>
#include <QJsonDocument>
#include <QtCharts>

using namespace QtCharts;

static const char* config_file = "config_bars.json";
static const bool force_config = false;


PlotterBarChart::PlotterBarChart(const BenchResults &bchResults, const QVector<int> &bchIdxs,
                                 const PlotParams &plotParams, const QString &origFilename,
                                 const QVector<FileReload>& addFilenames, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::PlotterBarChart)
    , mBenchIdxs(bchIdxs)
    , mPlotParams(plotParams)
    , mOrigFilename(origFilename)
    , mAddFilenames(addFilenames)
    , mAllIndexes(bchIdxs.size() == bchResults.benchmarks.size())
    , mWatcher(parent)
    , mIsVert(plotParams.type == ChartBarType)
{
    // UI
    ui->setupUi(this);
    this->setAttribute(Qt::WA_DeleteOnClose);
    
    QFileInfo fileInfo(origFilename);
    QString chartType = mIsVert ? "Bars - " : "HBars - ";
    this->setWindowTitle(chartType + fileInfo.fileName());
    
    connectUI();
    
    // Init
    setupChart(bchResults, bchIdxs, plotParams);
    setupOptions();
    
    // Show
    ui->horizontalLayout->insertWidget(0, mChartView);
}

PlotterBarChart::~PlotterBarChart()
{
    // Save options to file
    saveConfig();
    
    delete ui;
}

void PlotterBarChart::connectUI()
{
    // Theme
    ui->comboBoxTheme->addItem("Light",         QChart::ChartThemeLight);
    ui->comboBoxTheme->addItem("Blue Cerulean", QChart::ChartThemeBlueCerulean);
    ui->comboBoxTheme->addItem("Dark",          QChart::ChartThemeDark);
    ui->comboBoxTheme->addItem("Brown Sand",    QChart::ChartThemeBrownSand);
    ui->comboBoxTheme->addItem("Blue Ncs",      QChart::ChartThemeBlueNcs);
    ui->comboBoxTheme->addItem("High Contrast", QChart::ChartThemeHighContrast);
    ui->comboBoxTheme->addItem("Blue Icy",      QChart::ChartThemeBlueIcy);
    ui->comboBoxTheme->addItem("Qt",            QChart::ChartThemeQt);
    connect(ui->comboBoxTheme, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboThemeChanged);
    
    // Legend
    connect(ui->checkBoxLegendVisible, &QCheckBox::stateChanged, this, &PlotterBarChart::onCheckLegendVisible);
    
    ui->comboBoxLegendAlign->addItem("Top",     Qt::AlignTop);
    ui->comboBoxLegendAlign->addItem("Bottom",  Qt::AlignBottom);
    ui->comboBoxLegendAlign->addItem("Left",    Qt::AlignLeft);
    ui->comboBoxLegendAlign->addItem("Right",   Qt::AlignRight);
    connect(ui->comboBoxLegendAlign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboLegendAlignChanged);
    
    connect(ui->spinBoxLegendFontSize, QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterBarChart::onSpinLegendFontSizeChanged);
    connect(ui->pushButtonSeries, &QPushButton::clicked, this, &PlotterBarChart::onSeriesEditClicked);
    
    if (!isYTimeBased(mPlotParams.yType))
        ui->comboBoxTimeUnit->setEnabled(false);
    else
    {
        ui->comboBoxTimeUnit->addItem("ns", 1000.);
        ui->comboBoxTimeUnit->addItem("us", 1.);
        ui->comboBoxTimeUnit->addItem("ms", 0.001);
        connect(ui->comboBoxTimeUnit, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboTimeUnitChanged);
    }
    
    // Axes
    ui->comboBoxAxis->addItem("X-Axis");
    ui->comboBoxAxis->addItem("Y-Axis");
    connect(ui->comboBoxAxis, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboAxisChanged);
    
    ui->comboBoxValuePosition->addItem("None",       -1);
    ui->comboBoxValuePosition->addItem("Center",     QAbstractBarSeries::LabelsCenter);
    ui->comboBoxValuePosition->addItem("InsideEnd",  QAbstractBarSeries::LabelsInsideEnd);
    ui->comboBoxValuePosition->addItem("InsideBase", QAbstractBarSeries::LabelsInsideBase);
    ui->comboBoxValuePosition->addItem("OutsideEnd", QAbstractBarSeries::LabelsOutsideEnd);
    connect(ui->comboBoxValuePosition, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboValuePositionChanged);
    
    ui->comboBoxValueAngle->addItem("Right",  360.);
    ui->comboBoxValueAngle->addItem("Up",     -90.);
    ui->comboBoxValueAngle->addItem("Down",    90.);
    connect(ui->comboBoxValueAngle, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboValueAngleChanged);
    
    connect(ui->checkBoxAxisVisible, &QCheckBox::stateChanged, this, &PlotterBarChart::onCheckAxisVisible);
    connect(ui->checkBoxTitle,       &QCheckBox::stateChanged, this, &PlotterBarChart::onCheckTitleVisible);
    connect(ui->checkBoxLog,         &QCheckBox::stateChanged, this, &PlotterBarChart::onCheckLog);
    connect(ui->spinBoxLogBase,      QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterBarChart::onSpinLogBaseChanged);
    connect(ui->lineEditTitle,       &QLineEdit::textChanged, this, &PlotterBarChart::onEditTitleChanged);
    connect(ui->spinBoxTitleSize,    QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterBarChart::onSpinTitleSizeChanged);
    connect(ui->lineEditFormat,      &QLineEdit::textChanged, this, &PlotterBarChart::onEditFormatChanged);
    connect(ui->spinBoxLabelSize,    QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterBarChart::onSpinLabelSizeChanged);
    connect(ui->doubleSpinBoxMin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PlotterBarChart::onSpinMinChanged);
    connect(ui->doubleSpinBoxMax,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PlotterBarChart::onSpinMaxChanged);
    connect(ui->comboBoxMin,         QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboMinChanged);
    connect(ui->comboBoxMax,         QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterBarChart::onComboMaxChanged);
    connect(ui->spinBoxTicks,        QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterBarChart::onSpinTicksChanged);
    connect(ui->spinBoxMTicks,       QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterBarChart::onSpinMTicksChanged);
    
    // Actions
    connect(&mWatcher,              &QFileSystemWatcher::fileChanged, this, &PlotterBarChart::onAutoReload);
    connect(ui->checkBoxAutoReload, &QCheckBox::stateChanged, this, &PlotterBarChart::onCheckAutoReload);
    connect(ui->pushButtonReload,   &QPushButton::clicked, this, &PlotterBarChart::onReloadClicked);
    connect(ui->pushButtonSnapshot, &QPushButton::clicked, this, &PlotterBarChart::onSnapshotClicked);
}

void PlotterBarChart::setupChart(const BenchResults &bchResults, const QVector<int> &bchIdxs, const PlotParams &plotParams, bool init)
{
    QScopedPointer<QChart> scopedChart;
    QChart* chart = nullptr;
    if (init) {
        scopedChart.reset( new QChart() );
        chart = scopedChart.get();
    }
    else {  // Re-init
        chart = mChartView->chart();
        chart->setTitle("");
        chart->removeAllSeries();
        const auto xAxes = chart->axes(Qt::Horizontal);
        if ( !xAxes.empty() )
            chart->removeAxis( xAxes.constFirst() );
        const auto yAxes = chart->axes(Qt::Vertical);
        if ( !yAxes.empty() )
            chart->removeAxis( yAxes.constFirst() );
        mSeriesMapping.clear();
    }
    Q_ASSERT(chart);
    
    // Time unit
    mCurrentTimeFactor = 1.;
    if ( isYTimeBased(mPlotParams.yType) ) {
        if (     bchResults.meta.time_unit == "ns") mCurrentTimeFactor = 1000.;
        else if (bchResults.meta.time_unit == "ms") mCurrentTimeFactor = 0.001;
    }
    
    // Single series, one barset per benchmark type
    QScopedPointer<QAbstractBarSeries> scopedSeries;
    if (mIsVert)    scopedSeries.reset(new QBarSeries());
    else            scopedSeries.reset(new QHorizontalBarSeries());
    QAbstractBarSeries* series = scopedSeries.get();
    
    
    // 2D Bars
    // X: argumentA or templateB
    // Y: time/iter/bytes/items (not name dependent)
    // Bar: one per benchmark % X-param
    QVector<BenchSubset> bchSubsets = bchResults.groupParam(plotParams.xType == PlotArgumentType,
                                                            bchIdxs, plotParams.xIdx, "X");
    // Ignore empty series
    if ( bchSubsets.isEmpty() ) {
        qWarning() << "No compatible series to display";
    }
    
    bool firstCol = true;
    QStringList prevColLabels;
    for (const auto& bchSubset : qAsConst(bchSubsets))
    {
        // Ignore empty set
        if ( bchSubset.idxs.isEmpty() ) {
            qWarning() << "No X-value to trace bar for:" << bchSubset.name;
            continue;
        }
        
        const QString& subsetName = bchSubset.name;
//        qDebug() << "subsetName:" << subsetName;
//        qDebug() << "subsetIdxs:" << bchSubset.idxs;
        
        // X-row
        QScopedPointer<QBarSet> barSet(new QBarSet( subsetName.toHtmlEscaped() ));
        mSeriesMapping.push_back({subsetName, subsetName}); // color set later
        
        QStringList colLabels;
        for (int idx : bchSubset.idxs)
        {
            QString xName = bchResults.getParamName(plotParams.xType == PlotArgumentType,
                                                    idx, plotParams.xIdx);
            colLabels.append( xName.toHtmlEscaped() );
            
            // Add column
            barSet->append(getYPlotValue(bchResults.benchmarks[idx], plotParams.yType) * mCurrentTimeFactor);
        }
        // Add set (i.e. color)
        series->append(barSet.take());
        
        // Set column labels (only if no collision, empty otherwise)
        if (firstCol) // init
            prevColLabels = colLabels;
        else if ( commonPartEqual(prevColLabels, colLabels) ) {
            if (prevColLabels.size() < colLabels.size()) // replace by longest
                prevColLabels = colLabels;
        }
        else { // collision
            prevColLabels = QStringList("");
        }
        firstCol = false;
    }
    // Add the series
    chart->addSeries(scopedSeries.take());
    
    // Axes
    if (series->count() > 0)
    {
        // Chart type
        Qt::Alignment catAlign = mIsVert ? Qt::AlignBottom : Qt::AlignLeft;
        Qt::Alignment valAlign = mIsVert ? Qt::AlignLeft   : Qt::AlignBottom;
        
        // X-axis
        QBarCategoryAxis* catAxis = new QBarCategoryAxis();
        catAxis->append(prevColLabels);
        chart->addAxis(catAxis, catAlign);
        series->attachAxis(catAxis);
        if (plotParams.xType == PlotArgumentType)
            catAxis->setTitleText("Argument " + QString::number(plotParams.xIdx+1));
        else if (plotParams.xType == PlotTemplateType)
            catAxis->setTitleText("Template " + QString::number(plotParams.xIdx+1));
        
        // Y-axis
        QValueAxis* valAxis = new QValueAxis();
        chart->addAxis(valAxis, valAlign);
        series->attachAxis(valAxis);
        valAxis->applyNiceNumbers();
        valAxis->setTitleText( getYPlotName(plotParams.yType, bchResults.meta.time_unit) );
    }
    else
        chart->setTitle("No compatible series to display");
    
    if (init)
    {
        // View
        mChartView = new QChartView(scopedChart.take(), this);
        mChartView->setRenderHint(QPainter::Antialiasing);
    }
}

void PlotterBarChart::setupOptions(bool init)
{
    auto chart = mChartView->chart();
    
    // General
    if (init)
    {
        chart->setTheme(QChart::ChartThemeLight);
        chart->legend()->setAlignment(Qt::AlignTop);
        chart->legend()->setShowToolTips(true);
    }
    ui->spinBoxLegendFontSize->setValue( chart->legend()->font().pointSize() );
    
    mIgnoreEvents = true;
    int prevAxisIdx = ui->comboBoxAxis->currentIndex();
    
    if (!init)  // Re-init
    {
        mAxesParams[1].visible = true;
        mAxesParams[1].title = true;
        ui->comboBoxAxis->setCurrentIndex(0);
        ui->comboBoxMin->clear();
        ui->comboBoxMax->clear();
        ui->checkBoxAxisVisible->setChecked(true);
        ui->checkBoxTitle->setChecked(true);
        ui->checkBoxLog->setChecked(false);
        ui->comboBoxValuePosition->setCurrentIndex(0);
        ui->comboBoxValueAngle->setCurrentIndex(0);
    }
    
    // Time unit
    if      (mCurrentTimeFactor > 1.) ui->comboBoxTimeUnit->setCurrentIndex(0); // ns
    else if (mCurrentTimeFactor < 1.) ui->comboBoxTimeUnit->setCurrentIndex(2); // ms
    else                              ui->comboBoxTimeUnit->setCurrentIndex(1); // us
    
    // Axes
    Qt::Orientation xOrient = mIsVert ? Qt::Horizontal : Qt::Vertical;
    const auto& xAxes = chart->axes(xOrient);
    if ( !xAxes.isEmpty() )
    {
        QBarCategoryAxis* xAxis = (QBarCategoryAxis*)(xAxes.first());
        auto& axisParam = mAxesParams[0];
        
        axisParam.titleText = xAxis->titleText();
        axisParam.titleSize = xAxis->titleFont().pointSize();
        axisParam.labelSize = xAxis->labelsFont().pointSize();
        
        ui->labelFormat->setVisible(false);
        ui->lineEditFormat->setVisible(false);
        ui->doubleSpinBoxMin->setVisible(false);
        ui->doubleSpinBoxMax->setVisible(false);
        
        ui->lineEditTitle->setText( axisParam.titleText );
        ui->lineEditTitle->setCursorPosition(0);
        ui->spinBoxTitleSize->setValue( axisParam.titleSize );
        ui->spinBoxLabelSize->setValue( axisParam.labelSize );
        const auto categories = xAxis->categories();
        for (const auto& cat : categories) {
            ui->comboBoxMin->addItem(cat);
            ui->comboBoxMax->addItem(cat);
        }
        ui->comboBoxMax->setCurrentIndex( ui->comboBoxMax->count()-1 );
        
    }
    Qt::Orientation yOrient = mIsVert ? Qt::Vertical : Qt::Horizontal;
    const auto& yAxes = chart->axes(yOrient);
    if ( !yAxes.isEmpty() )
    {
        QValueAxis* yAxis = (QValueAxis*)(yAxes.first());
        auto& axisParam = mAxesParams[1];
        
        axisParam.titleText = yAxis->titleText();
        axisParam.titleSize = yAxis->titleFont().pointSize();
        axisParam.labelSize = yAxis->labelsFont().pointSize();
        
        ui->lineEditFormat->setText( "%g" );
        ui->lineEditFormat->setCursorPosition(0);
        yAxis->setLabelFormat( ui->lineEditFormat->text() );
        ui->doubleSpinBoxMin->setValue( yAxis->min() );
        ui->doubleSpinBoxMax->setValue( yAxis->max() );
        ui->spinBoxTicks->setValue( yAxis->tickCount() );
        ui->spinBoxMTicks->setValue( yAxis->minorTickCount() );
    }
    mIgnoreEvents = false;
    
    
    // Load options from file
    loadConfig(init);
    
    
    // Apply actions
    if (ui->checkBoxAutoReload->isChecked())
        onCheckAutoReload(Qt::Checked);
    
    // Update series color config
    if (!chart->series().empty())
    {
        const auto barSeries = (QAbstractBarSeries*)chart->series().at(0);
        for (int idx = 0 ; idx < mSeriesMapping.size(); ++idx)
        {
            auto& config = mSeriesMapping[idx];
            const auto& barSet = barSeries->barSets().at(idx);
            
            config.oldColor = barSet->color();
            if (!config.newColor.isValid())
                config.newColor = barSet->color();  // init
            else
                barSet->setColor(config.newColor);  // apply
            
            if (config.newName != config.oldName)
                barSet->setLabel( config.newName.toHtmlEscaped() );
        }
    }
    
    // Restore selected axis
    if (!init)
        ui->comboBoxAxis->setCurrentIndex(prevAxisIdx);
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() + ")");
}

void PlotterBarChart::loadConfig(bool init)
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
        
        // Legend
        if (json.contains("legend.visible") && json["legend.visible"].isBool())
            ui->checkBoxLegendVisible->setChecked( json["legend.visible"].toBool() );
        if (json.contains("legend.align") && json["legend.align"].isString())
            ui->comboBoxLegendAlign->setCurrentText( json["legend.align"].toString() );
        if (json.contains("legend.fontSize") && json["legend.fontSize"].isDouble())
            ui->spinBoxLegendFontSize->setValue( json["legend.fontSize"].toInt(8) );
        
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
        for (int idx = 0; idx < 2; ++idx)
        {
            auto& axis = mAxesParams[idx];
            
            if (json.contains(prefix + ".visible") && json[prefix + ".visible"].isBool()) {
                axis.visible = json[prefix + ".visible"].toBool();
                ui->checkBoxAxisVisible->setChecked( axis.visible );
            }
            if (json.contains(prefix + ".title") && json[prefix + ".title"].isBool()) {
                axis.title = json[prefix + ".title"].toBool();
                ui->checkBoxTitle->setChecked( axis.title );
            }
            if (json.contains(prefix + ".titleSize") && json[prefix + ".titleSize"].isDouble()) {
                axis.titleSize = json[prefix + ".titleSize"].toInt(8);
                ui->spinBoxTitleSize->setValue( axis.titleSize );
            }
            if (json.contains(prefix + ".labelSize") && json[prefix + ".labelSize"].isDouble()) {
                axis.labelSize = json[prefix + ".labelSize"].toInt(8);
                ui->spinBoxLabelSize->setValue( axis.labelSize );
            }
            if (!init)
            {
                if (json.contains(prefix + ".titleText") && json[prefix + ".titleText"].isString()) {
                    axis.titleText = json[prefix + ".titleText"].toString();
                    ui->lineEditTitle->setText( axis.titleText );
                    ui->lineEditTitle->setCursorPosition(0);
                }
            }
            // x-axis
            if (idx == 0)
            {
                if (json.contains(prefix + ".value.position") && json[prefix + ".value.position"].isString())
                    ui->comboBoxValuePosition->setCurrentText( json[prefix + ".value.position"].toString() );
                if (json.contains(prefix + ".value.angle") && json[prefix + ".value.angle"].isString())
                    ui->comboBoxValueAngle->setCurrentText( json[prefix + ".value.angle"].toString() );
                if (force_config)
                {
                    if (json.contains(prefix + ".min") && json[prefix + ".min"].isString())
                        ui->comboBoxMin->setCurrentText( json[prefix + ".min"].toString() );
                    if (json.contains(prefix + ".max") && json[prefix + ".max"].isString())
                        ui->comboBoxMax->setCurrentText( json[prefix + ".max"].toString() );
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
                    ui->spinBoxMTicks->setValue( json[prefix + ".mticks"].toInt(0) );
                if (!init)
                {
                    if (json.contains(prefix + ".min") && json[prefix + ".min"].isDouble())
                        ui->doubleSpinBoxMin->setValue( json[prefix + ".min"].toDouble() );
                    if (json.contains(prefix + ".max") && json[prefix + ".max"].isDouble())
                        ui->doubleSpinBoxMax->setValue( json[prefix + ".max"].toDouble() );
                }
            }
            
            prefix = "axis.y";
            ui->comboBoxAxis->setCurrentIndex(1);
        }
        ui->comboBoxAxis->setCurrentIndex(0);
    }
    else
    {
        if (configFile.exists())
            qWarning() << "Couldn't read: " << QString(config_folder) + config_file;
    }
}

void PlotterBarChart::saveConfig()
{
    QFile configFile(QString(config_folder) + config_file);
    if (configFile.open(QIODevice::WriteOnly))
    {
        QJsonObject json;
        
        // Theme
        json["theme"] = ui->comboBoxTheme->currentText();
        // Legend
        json["legend.visible"]  = ui->checkBoxLegendVisible->isChecked();
        json["legend.align"]    = ui->comboBoxLegendAlign->currentText();
        json["legend.fontSize"] = ui->spinBoxLegendFontSize->value();
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
        for (int idx = 0; idx < 2; ++idx)
        {
            const auto& axis = mAxesParams[idx];
            
            json[prefix + ".visible"]   = axis.visible;
            json[prefix + ".title"]     = axis.title;
            json[prefix + ".titleText"] = axis.titleText;
            json[prefix + ".titleSize"] = axis.titleSize;
            json[prefix + ".labelSize"] = axis.labelSize;
            // x-axis
            if (idx == 0)
            {
                json[prefix + ".value.position"] = ui->comboBoxValuePosition->currentText();
                json[prefix + ".value.angle"]    = ui->comboBoxValueAngle->currentText();
                json[prefix + ".min"]            = ui->comboBoxMin->currentText();
                json[prefix + ".max"]            = ui->comboBoxMax->currentText();
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
            }
            prefix = "axis.y";
        }
        
        configFile.write( QJsonDocument(json).toJson() );
    }
    else
        qWarning() << "Couldn't update: " << QString(config_folder) + config_file;
}

//
// Theme
void PlotterBarChart::onComboThemeChanged(int index)
{
    QChart::ChartTheme theme = static_cast<QChart::ChartTheme>(
                ui->comboBoxTheme->itemData(index).toInt());
    mChartView->chart()->setTheme(theme);
    
    // Update series color
    const auto& series = mChartView->chart()->series();
    if (!series.empty())
    {
        const auto barSeries = (QAbstractBarSeries*)series.at(0);
        for (int idx = 0 ; idx < mSeriesMapping.size(); ++idx)
        {
            auto& config = mSeriesMapping[idx];
            const auto& barSet = barSeries->barSets().at(idx);
            auto prevColor = config.oldColor;
            
            config.oldColor = barSet->color();
            if (config.newColor != prevColor)
                barSet->setColor(config.newColor); // re-apply config
            else
                config.newColor = config.oldColor; // sync with theme
        }
    }
    
    // Re-apply font sizes
    onSpinLegendFontSizeChanged( ui->spinBoxLegendFontSize->value() );
    onSpinLabelSizeChanged2(mAxesParams[0].labelSize, 0);
    onSpinLabelSizeChanged2(mAxesParams[1].labelSize, 1);
    onSpinTitleSizeChanged2(mAxesParams[0].titleSize, 0);
    onSpinTitleSizeChanged2(mAxesParams[1].titleSize, 1);
}

//
// Legend
void PlotterBarChart::onCheckLegendVisible(int state)
{
    mChartView->chart()->legend()->setVisible(state == Qt::Checked);
}

void PlotterBarChart::onComboLegendAlignChanged(int index)
{
    Qt::Alignment align = static_cast<Qt::Alignment>(
                ui->comboBoxLegendAlign->itemData(index).toInt());
    mChartView->chart()->legend()->setAlignment(align);
}

void PlotterBarChart::onSpinLegendFontSizeChanged(int i)
{
    QFont font = mChartView->chart()->legend()->font();
    font.setPointSize(i);
    mChartView->chart()->legend()->setFont(font);
}

void PlotterBarChart::onSeriesEditClicked()
{
    SeriesDialog seriesDialog(mSeriesMapping, this);
    auto res = seriesDialog.exec();
    const auto& series = mChartView->chart()->series();
    if (res == QDialog::Accepted && !series.empty())
    {
        const auto barSeries = (QAbstractBarSeries*)series.at(0);
        const auto& newMapping = seriesDialog.getMapping();
        for (int idx = 0; idx < newMapping.size(); ++idx)
        {
            const auto& newPair = newMapping[idx];
            const auto& oldPair = mSeriesMapping[idx];
            const auto& barSet = barSeries->barSets().at(idx);
            
            if (newPair.newName != oldPair.newName) {
                barSet->setLabel( newPair.newName.toHtmlEscaped() );
            }
            if (newPair.newColor != oldPair.newColor) {
                barSet->setColor(newPair.newColor);
            }
        }
        mSeriesMapping = newMapping;
    }
}

void PlotterBarChart::onComboTimeUnitChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    
    // Update data
    double unitFactor = ui->comboBoxTimeUnit->currentData().toDouble();
    double updateFactor = unitFactor / mCurrentTimeFactor;  // can cause precision loss
    auto chartSeries = mChartView->chart()->series();
    if (chartSeries.empty())
        return;
    
    const QAbstractBarSeries* barSeries = (QAbstractBarSeries*)chartSeries[0];
    auto barSets = barSeries->barSets();
    for (int idx = 0; idx < barSets.size(); ++idx)
    {
        auto* barSet = barSets.at(idx);
        for (int i = 0; i < barSet->count(); ++i) {
            qreal val = barSet->at(i);
            barSet->replace(i, val * updateFactor);
        }
    }
    
    // Update axis title
    QString oldUnitName = "(us)";
    if      (mCurrentTimeFactor > 1.) oldUnitName = "(ns)";
    else if (mCurrentTimeFactor < 1.) oldUnitName = "(ms)";
    
    Qt::Orientation yOrient = mIsVert ? Qt::Vertical : Qt::Horizontal;
    const auto& axes = mChartView->chart()->axes(yOrient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        QString axisTitle = axis->titleText();
        if (axisTitle.endsWith(oldUnitName)) {
            QString unitName  = ui->comboBoxTimeUnit->currentText();
            onEditTitleChanged2(axisTitle.replace(axisTitle.size() - 3, 2, unitName), 1);
        }
    }
    // Update range
    ui->doubleSpinBoxMin->setValue(ui->doubleSpinBoxMin->value() * updateFactor);
    ui->doubleSpinBoxMax->setValue(ui->doubleSpinBoxMax->value() * updateFactor);
    if (ui->comboBoxAxis->currentIndex() != 1 && !axes.isEmpty())
    {
        QValueAxis* yAxis = (QValueAxis*)(axes.first());
        onSpinMinChanged2(yAxis->min() * updateFactor, 1);
        onSpinMaxChanged2(yAxis->max() * updateFactor, 1);
    }
    
    mCurrentTimeFactor = unitFactor;
}

//
// Axes
void PlotterBarChart::onComboAxisChanged(int idx)
{
    // Update UI
    bool wasIgnoring = mIgnoreEvents;
    mIgnoreEvents = true;
    
    ui->checkBoxAxisVisible->setChecked( mAxesParams[idx].visible );
    ui->checkBoxTitle->setChecked( mAxesParams[idx].title );
    ui->checkBoxLog->setEnabled( idx == 1 );
    ui->spinBoxLogBase->setEnabled( ui->checkBoxLog->isEnabled() && ui->checkBoxLog->isChecked() );
    ui->lineEditTitle->setText( mAxesParams[idx].titleText );
    ui->lineEditTitle->setCursorPosition(0);
    ui->spinBoxTitleSize->setValue( mAxesParams[idx].titleSize );
    ui->labelFormat->setVisible( idx == 1 );
    ui->lineEditFormat->setVisible( idx == 1 );
    ui->labelValue->setVisible( idx == 0 );
    ui->comboBoxValuePosition->setVisible( idx == 0 );
    ui->comboBoxValueAngle->setVisible( idx == 0 );
    ui->spinBoxLabelSize->setValue( mAxesParams[idx].labelSize );
    ui->comboBoxMin->setVisible( idx == 0 );
    ui->comboBoxMax->setVisible( idx == 0 );
    ui->doubleSpinBoxMin->setVisible( idx == 1 );
    ui->doubleSpinBoxMax->setVisible( idx == 1 );
    ui->spinBoxTicks->setEnabled(   idx == 1 && !ui->checkBoxLog->isChecked() );
    ui->spinBoxMTicks->setEnabled(  idx == 1 );
    
    mIgnoreEvents = wasIgnoring;
}

void PlotterBarChart::onCheckAxisVisible(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setVisible(state == Qt::Checked);
        mAxesParams[iAxis].visible = state == Qt::Checked;
    }
}

void PlotterBarChart::onCheckTitleVisible(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setTitleVisible(state == Qt::Checked);
        mAxesParams[iAxis].title = state == Qt::Checked;
    }
}

void PlotterBarChart::onCheckLog(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal  : Qt::Vertical;
    Qt::Alignment   align  = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::AlignBottom : Qt::AlignLeft;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() )
    {
        if (state == Qt::Checked)
        {
            QValueAxis* axis = (QValueAxis*)(axes.first());
    
            QLogValueAxis* logAxis = new QLogValueAxis();
            logAxis->setVisible( axis->isVisible() );
            logAxis->setTitleVisible( axis->isTitleVisible() );
            logAxis->setTitleText( axis->titleText() );
            logAxis->setTitleFont( axis->titleFont() );
            logAxis->setLabelFormat( axis->labelFormat() );
            logAxis->setLabelsFont( axis->labelsFont() );
            
            mChartView->chart()->removeAxis(axis);
            mChartView->chart()->addAxis(logAxis, align);
            const auto series = mChartView->chart()->series();
            for (const auto& series : series)
                series->attachAxis(logAxis);
            
            logAxis->setBase( ui->spinBoxLogBase->value() );
            logAxis->setMin( ui->doubleSpinBoxMin->value() );
            logAxis->setMax( ui->doubleSpinBoxMax->value() );
            logAxis->setMinorTickCount( ui->spinBoxMTicks->value() );
        }
        else
        {
            QLogValueAxis* logAxis = (QLogValueAxis*)(axes.first());
    
            QValueAxis* axis = new QValueAxis();
            axis->setVisible( logAxis->isVisible() );
            axis->setTitleVisible( logAxis->isTitleVisible() );
            axis->setTitleText( logAxis->titleText() );
            axis->setTitleFont( logAxis->titleFont() );
            axis->setLabelFormat( logAxis->labelFormat() );
            axis->setLabelsFont( logAxis->labelsFont() );
            
            mChartView->chart()->removeAxis(logAxis);
            mChartView->chart()->addAxis(axis, align);
            const auto chartSeries = mChartView->chart()->series();
            for (const auto& series : chartSeries)
                series->attachAxis(axis);
            
            axis->setMin( ui->doubleSpinBoxMin->value() );
            axis->setMax( ui->doubleSpinBoxMax->value() );
            axis->setTickCount( ui->spinBoxTicks->value() );
            axis->setMinorTickCount( ui->spinBoxMTicks->value() );
        }
        ui->spinBoxTicks->setEnabled(  state != Qt::Checked);
        ui->spinBoxLogBase->setEnabled(state == Qt::Checked);
    }
}

void PlotterBarChart::onSpinLogBaseChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() && ui->checkBoxLog->isChecked())
    {
        QLogValueAxis* logAxis = (QLogValueAxis*)(axes.first());
        logAxis->setBase(i);
    }
}

void PlotterBarChart::onEditTitleChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onEditTitleChanged2(text, iAxis);
}

void PlotterBarChart::onEditTitleChanged2(const QString& text, int iAxis)
{
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setTitleText(text);
        mAxesParams[iAxis].titleText = text;
    }
}

void PlotterBarChart::onSpinTitleSizeChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinTitleSizeChanged2(i, iAxis);
}

void PlotterBarChart::onSpinTitleSizeChanged2(int i, int iAxis)
{
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        
        QFont font = axis->titleFont();
        font.setPointSize(i);
        axis->setTitleFont(font);
        mAxesParams[iAxis].titleSize = i;
    }
}

void PlotterBarChart::onEditFormatChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() )
    {
        if ( !ui->checkBoxLog->isChecked() ) {
            QValueAxis* axis = (QValueAxis*)(axes.first());
            axis->setLabelFormat(text);
        }
        else {
            QLogValueAxis* axis = (QLogValueAxis*)(axes.first());
            axis->setLabelFormat(text);
        }
    }
}

void PlotterBarChart::onComboValuePositionChanged(int index)
{
    if (mIgnoreEvents) return;
    const auto chartSeries = mChartView->chart()->series();
    for (auto series : chartSeries)
    {
        QAbstractBarSeries* barSeries = (QAbstractBarSeries*)(series);
        if (index == 0)
            barSeries->setLabelsVisible(false);
        else {
            barSeries->setLabelsVisible(true);
            barSeries->setLabelsPosition( (QAbstractBarSeries::LabelsPosition)
                                          (ui->comboBoxValuePosition->currentData().toInt()) );
        }
    }
}

void PlotterBarChart::onComboValueAngleChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    const auto chartSeries = mChartView->chart()->series();
    for (auto series : chartSeries)
    {
        QAbstractBarSeries* barSeries = (QAbstractBarSeries*)(series);
        barSeries->setLabelsAngle( ui->comboBoxValueAngle->currentData().toDouble() );
    }
}

void PlotterBarChart::onSpinLabelSizeChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinLabelSizeChanged2(i, iAxis);
}

void PlotterBarChart::onSpinLabelSizeChanged2(int i, int iAxis)
{
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        
        QFont font = axis->labelsFont();
        font.setPointSize(i);
        axis->setLabelsFont(font);
        mAxesParams[iAxis].labelSize = i;
    }
}

void PlotterBarChart::onSpinMinChanged(double d)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinMinChanged2(d, iAxis);
}

void PlotterBarChart::onSpinMinChanged2(double d, int iAxis)
{
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setMin(d);
    }
}

void PlotterBarChart::onSpinMaxChanged(double d)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinMaxChanged2(d, iAxis);
}

void PlotterBarChart::onSpinMaxChanged2(double d, int iAxis)
{
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setMax(d);
    }
}

void PlotterBarChart::onComboMinChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QBarCategoryAxis* axis = (QBarCategoryAxis*)(axes.first());
        axis->setMin( ui->comboBoxMin->currentText() );
    }
}

void PlotterBarChart::onComboMaxChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QBarCategoryAxis* axis = (QBarCategoryAxis*)(axes.first());
        axis->setMax( ui->comboBoxMax->currentText() );
    }
}

void PlotterBarChart::onSpinTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() )
    {
        if ( !ui->checkBoxLog->isChecked() ) {
            QValueAxis* axis = (QValueAxis*)(axes.first());
            axis->setTickCount(i);
        }
    }
}

void PlotterBarChart::onSpinMTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = (iAxis == 0 && mIsVert) || (iAxis == 1 && !mIsVert)
            ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() )
    {
        if ( !ui->checkBoxLog->isChecked() ) {
            QValueAxis* axis = (QValueAxis*)(axes.first());
            axis->setMinorTickCount(i);
        }
        else {
            QLogValueAxis* axis = (QLogValueAxis*)(axes.first());
            axis->setMinorTickCount(i);
            
            // Force update
            const int base = (int)axis->base();
            axis->setBase(base + 1);
            axis->setBase(base);
        }
    }
}

//
// Actions
void PlotterBarChart::onCheckAutoReload(int state)
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

void PlotterBarChart::onAutoReload(const QString &path)
{
    QFileInfo fi(path);
    if (fi.exists() && fi.isReadable() && fi.size() > 0)
        onReloadClicked();
    else
        qWarning() << "Unable to auto-reload file: " << path;
}

void PlotterBarChart::onReloadClicked()
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
    
    QVector<BenchSubset> newBchSubsets = newBchResults.groupParam(mPlotParams.xType == PlotArgumentType,
                                                                  mBenchIdxs, mPlotParams.xIdx, "X");
    int newBarSetIdx = 0;
    const auto& oldChartSeries = mChartView->chart()->series();
    if ( newBchSubsets.isEmpty() ) {
        errorMsg = "No compatible series to display";   // Ignore empty series
    }
    if (oldChartSeries.size() != 1) {
        errorMsg = "No compatible series to display originally";
    }
    
    if (errorMsg.isEmpty())
    {
        const QAbstractBarSeries* oldBarSeries = (QAbstractBarSeries*)oldChartSeries[0];
        for (const auto& bchSubset : qAsConst(newBchSubsets))
        {
            // Ignore empty set
            if ( bchSubset.idxs.isEmpty() )
                continue;
            if (newBarSetIdx >= oldBarSeries->count())
                break;
            
            const QString& subsetName = bchSubset.name;
            if (subsetName != mSeriesMapping[newBarSetIdx].oldName) {
                errorMsg = "Series has different name";
                break;
            }
            const auto barSet = oldBarSeries->barSets().at(newBarSetIdx);
            if (bchSubset.idxs.size() != barSet->count())
            {
                errorMsg = "Number of series bars is different";
                break;
            }
            ++newBarSetIdx;
        }
        if (newBarSetIdx != oldBarSeries->count()) {
            errorMsg = "Number of series is different";
        }
    }
    
    // Direct update if compatible
    if ( errorMsg.isEmpty() )
    {
        newBarSetIdx = 0;
        const QAbstractBarSeries* oldBarSeries = (QAbstractBarSeries*)oldChartSeries[0];
        for (const auto& bchSubset : qAsConst(newBchSubsets))
        {
            // Ignore empty set
            if ( bchSubset.idxs.isEmpty() ) {
                qWarning() << "No X-value to trace bar for:" << bchSubset.name;
                continue;
            }
            
            // Update points
            const auto& barSet = oldBarSeries->barSets().at(newBarSetIdx);
            barSet->remove(0, barSet->count());
            
            for (int idx : bchSubset.idxs) {
                // Add column
                barSet->append(getYPlotValue(newBchResults.benchmarks[idx], mPlotParams.yType) * mCurrentTimeFactor);
            }
            ++newBarSetIdx;
        }
    }
    // Reset update if all benchmarks
    else if (mAllIndexes)
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
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() +")");
}

void PlotterBarChart::onSnapshotClicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("Save snapshot"), "", tr("Images (*.png)"));
    
    if ( !fileName.isEmpty() )
    {
        QPixmap pixmap = mChartView->grab();
        
        bool ok = pixmap.save(fileName, "PNG");
        if (!ok)
            QMessageBox::warning(this, "Chart snapshot", "Error saving snapshot file.");
    }
}
