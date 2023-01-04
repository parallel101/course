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

#include "plotter_linechart.h"
#include "ui_plotter_linechart.h"

#include "benchmark_results.h"
#include "result_parser.h"

#include <QFileInfo>
#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QtCharts>

using namespace QtCharts;

static const char* config_file = "config_lines.json";


PlotterLineChart::PlotterLineChart(const BenchResults &bchResults, const QVector<int> &bchIdxs,
                                   const PlotParams &plotParams, const QString &origFilename,
                                   const QVector<FileReload>& addFilenames, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::PlotterLineChart)
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
    QString chartType = (plotParams.type == ChartLineType) ? "Lines - " : "Splines - ";
    this->setWindowTitle(chartType + fileInfo.fileName());
    
    connectUI();
    
    //TODO: select points
    //See: https://doc.qt.io/qt-5/qtcharts-callout-example.html
    
    // Init
    setupChart(bchResults, bchIdxs, plotParams);
    setupOptions();
    
    // Show
    ui->horizontalLayout->insertWidget(0, mChartView);
}

PlotterLineChart::~PlotterLineChart()
{
    // Save options to file
    saveConfig();
    
    delete ui;
}

void PlotterLineChart::connectUI()
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
    connect(ui->comboBoxTheme, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterLineChart::onComboThemeChanged);
    
    // Legend
    connect(ui->checkBoxLegendVisible, &QCheckBox::stateChanged, this, &PlotterLineChart::onCheckLegendVisible);
    
    ui->comboBoxLegendAlign->addItem("Top",     Qt::AlignTop);
    ui->comboBoxLegendAlign->addItem("Bottom",  Qt::AlignBottom);
    ui->comboBoxLegendAlign->addItem("Left",    Qt::AlignLeft);
    ui->comboBoxLegendAlign->addItem("Right",   Qt::AlignRight);
    connect(ui->comboBoxLegendAlign, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterLineChart::onComboLegendAlignChanged);
    
    connect(ui->spinBoxLegendFontSize, QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterLineChart::onSpinLegendFontSizeChanged);
    connect(ui->pushButtonSeries, &QPushButton::clicked, this, &PlotterLineChart::onSeriesEditClicked);
    
    if (!isYTimeBased(mPlotParams.yType))
        ui->comboBoxTimeUnit->setEnabled(false);
    else
    {
        ui->comboBoxTimeUnit->addItem("ns", 1000.);
        ui->comboBoxTimeUnit->addItem("us", 1.);
        ui->comboBoxTimeUnit->addItem("ms", 0.001);
        connect(ui->comboBoxTimeUnit, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterLineChart::onComboTimeUnitChanged);
    }
    
    // Axes
    ui->comboBoxAxis->addItem("X-Axis");
    ui->comboBoxAxis->addItem("Y-Axis");
    connect(ui->comboBoxAxis, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PlotterLineChart::onComboAxisChanged);
    
    connect(ui->checkBoxAxisVisible, &QCheckBox::stateChanged, this, &PlotterLineChart::onCheckAxisVisible);
    connect(ui->checkBoxTitle,       &QCheckBox::stateChanged, this, &PlotterLineChart::onCheckTitleVisible);
    connect(ui->checkBoxLog,         &QCheckBox::stateChanged, this, &PlotterLineChart::onCheckLog);
    connect(ui->spinBoxLogBase,      QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterLineChart::onSpinLogBaseChanged);
    connect(ui->lineEditTitle,       &QLineEdit::textChanged, this, &PlotterLineChart::onEditTitleChanged);
    connect(ui->spinBoxTitleSize,    QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterLineChart::onSpinTitleSizeChanged);
    connect(ui->lineEditFormat,      &QLineEdit::textChanged, this, &PlotterLineChart::onEditFormatChanged);
    connect(ui->spinBoxLabelSize,    QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterLineChart::onSpinLabelSizeChanged);
    connect(ui->doubleSpinBoxMin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PlotterLineChart::onSpinMinChanged);
    connect(ui->doubleSpinBoxMax,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &PlotterLineChart::onSpinMaxChanged);
    connect(ui->spinBoxTicks,        QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterLineChart::onSpinTicksChanged);
    connect(ui->spinBoxMTicks,       QOverload<int>::of(&QSpinBox::valueChanged), this, &PlotterLineChart::onSpinMTicksChanged);
    
    // Actions
    connect(&mWatcher,              &QFileSystemWatcher::fileChanged, this, &PlotterLineChart::onAutoReload);
    connect(ui->checkBoxAutoReload, &QCheckBox::stateChanged, this, &PlotterLineChart::onCheckAutoReload);
    connect(ui->pushButtonReload,   &QPushButton::clicked, this, &PlotterLineChart::onReloadClicked);
    connect(ui->pushButtonSnapshot, &QPushButton::clicked, this, &PlotterLineChart::onSnapshotClicked);
}

void PlotterLineChart::setupChart(const BenchResults &bchResults, const QVector<int> &bchIdxs, const PlotParams &plotParams, bool init)
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
    
    
    // 2D Lines
    // X: argumentA or templateB
    // Y: time/iter/bytes/items (not name dependent)
    // Line: one per benchmark % X-param
    QVector<BenchSubset> bchSubsets = bchResults.groupParam(plotParams.xType == PlotArgumentType,
                                                            bchIdxs, plotParams.xIdx, "X");
    bool custDataAxis = true;
    QString custDataName;
    for (const auto& bchSubset : qAsConst(bchSubsets))
    {
        // Ignore single point lines
        if (bchSubset.idxs.size() < 2) {
            qWarning() << "Not enough points to trace line for: " << bchSubset.name;
            continue;
        }
        
        // Chart type
        QScopedPointer<QLineSeries> series;
        if (plotParams.type == ChartLineType)   series.reset(new QLineSeries());
        else                                    series.reset(new QSplineSeries());
        
        const QString& subsetName = bchSubset.name;
//        qDebug() << "subsetName:" << subsetName;
//        qDebug() << "subsetIdxs:" << bchSubset.idxs;
        
        double xFallback = 0.;
        for (int idx : bchSubset.idxs)
        {
            QString xName = bchResults.getParamName(plotParams.xType == PlotArgumentType,
                                                    idx, plotParams.xIdx);
            double xVal = BenchResults::getParamValue(xName, custDataName, custDataAxis, xFallback);
            
            // Add point
            series->append(xVal, getYPlotValue(bchResults.benchmarks[idx], plotParams.yType) * mCurrentTimeFactor);
        }
        // Add series
        series->setName( subsetName.toHtmlEscaped() );
        mSeriesMapping.push_back({subsetName, subsetName}); // color set later
        chart->addSeries(series.take());
    }
    
    //
    // Axes
    if ( !chart->series().isEmpty() )
    {
        chart->createDefaultAxes();
        
        // X-axis
        QValueAxis* xAxis = (QValueAxis*)(chart->axes(Qt::Horizontal).constFirst());
        if (plotParams.xType == PlotArgumentType)
            xAxis->setTitleText("Argument " + QString::number(plotParams.xIdx+1));
        else { // template
            if ( !custDataName.isEmpty() )
                xAxis->setTitleText(custDataName);
            else
                xAxis->setTitleText("Template " + QString::number(plotParams.xIdx+1));
        }
        xAxis->setTickCount(9);
        
        // Y-axis
        QValueAxis* yAxis = (QValueAxis*)(chart->axes(Qt::Vertical).constFirst());
        yAxis->setTitleText( getYPlotName(plotParams.yType, bchResults.meta.time_unit) );
        yAxis->applyNiceNumbers();
    }
    else
        chart->setTitle("No series with at least 2 points to display");
    
    if (init)
    {
        // View
        mChartView = new QChartView(scopedChart.take(), this);
        mChartView->setRenderHint(QPainter::Antialiasing);
    }
}

void PlotterLineChart::setupOptions(bool init)
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
        mAxesParams[0].log = false;
        mAxesParams[1].log = false;
        ui->comboBoxAxis->setCurrentIndex(0);
        ui->checkBoxAxisVisible->setChecked(true);
        ui->checkBoxTitle->setChecked(true);
        ui->checkBoxLog->setChecked(false);
    }
    
    // Time unit
    if      (mCurrentTimeFactor > 1.) ui->comboBoxTimeUnit->setCurrentIndex(0); // ns
    else if (mCurrentTimeFactor < 1.) ui->comboBoxTimeUnit->setCurrentIndex(2); // ms
    else                              ui->comboBoxTimeUnit->setCurrentIndex(1); // us
    
    // Axes
    const auto& hAxes = chart->axes(Qt::Horizontal);
    if ( !hAxes.isEmpty() )
    {
        QValueAxis* xAxis = (QValueAxis*)(hAxes.first());
        auto& axisParam = mAxesParams[0];
        
        axisParam.titleText = xAxis->titleText();
        axisParam.titleSize = xAxis->titleFont().pointSize();
        axisParam.labelFormat = "%g";
        xAxis->setLabelFormat(axisParam.labelFormat);
        axisParam.labelSize   = xAxis->labelsFont().pointSize();
        axisParam.min = xAxis->min();
        axisParam.max = xAxis->max();
        axisParam.ticks  = xAxis->tickCount();
        axisParam.mticks = xAxis->minorTickCount();
        
        ui->lineEditTitle->setText( axisParam.titleText );
        ui->lineEditTitle->setCursorPosition(0);
        ui->spinBoxTitleSize->setValue( axisParam.titleSize );
        ui->lineEditFormat->setText( axisParam.labelFormat );
        ui->lineEditFormat->setCursorPosition(0);
        ui->spinBoxLabelSize->setValue( axisParam.labelSize );
        ui->doubleSpinBoxMin->setValue( axisParam.min );
        ui->doubleSpinBoxMax->setValue( axisParam.max );
        ui->spinBoxTicks->setValue( axisParam.ticks );
        ui->spinBoxMTicks->setValue( axisParam.mticks );
    }
    const auto& vAxes = chart->axes(Qt::Vertical);
    if ( !vAxes.isEmpty() )
    {
        QValueAxis* yAxis = (QValueAxis*)(vAxes.first());
        auto& axisParam = mAxesParams[1];
        
        axisParam.titleText = yAxis->titleText();
        axisParam.titleSize = yAxis->titleFont().pointSize();
        axisParam.labelFormat = "%g";
        yAxis->setLabelFormat(axisParam.labelFormat);
        axisParam.labelSize   = yAxis->labelsFont().pointSize();
        axisParam.min = yAxis->min();
        axisParam.max = yAxis->max();
        axisParam.ticks  = yAxis->tickCount();
        axisParam.mticks = yAxis->minorTickCount();
    }
    mIgnoreEvents = false;
    
    
    // Load options from file
    loadConfig(init);
    
    
    // Apply actions
    if (ui->checkBoxAutoReload->isChecked())
        onCheckAutoReload(Qt::Checked);
    
    // Update series color config
    const auto& chartSeries = chart->series();
    for (int idx = 0 ; idx < mSeriesMapping.size(); ++idx)
    {
        auto& config = mSeriesMapping[idx];
        const auto& series = (QXYSeries*)chartSeries.at(idx);
        
        config.oldColor = series->color();
        if (!config.newColor.isValid())
            config.newColor = series->color();  // init
        else
            series->setColor(config.newColor);  // apply
        
        if (config.newName != config.oldName)
            series->setName( config.newName.toHtmlEscaped() );
    }
    
    // Restore selected axis
    if (!init)
        ui->comboBoxAxis->setCurrentIndex(prevAxisIdx);
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() + ")");
}

void PlotterLineChart::loadConfig(bool init)
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
            if (json.contains(prefix + ".log") && json[prefix + ".log"].isBool()) {
                axis.log = json[prefix + ".log"].toBool();
                ui->checkBoxLog->setChecked( axis.log );
            }
            if (json.contains(prefix + ".logBase") && json[prefix + ".logBase"].isDouble()) {
                axis.logBase = json[prefix + ".logBase"].toInt(10);
                ui->spinBoxLogBase->setValue( axis.logBase );
            }
            if (json.contains(prefix + ".titleSize") && json[prefix + ".titleSize"].isDouble()) {
                axis.titleSize = json[prefix + ".titleSize"].toInt(8);
                ui->spinBoxTitleSize->setValue( axis.titleSize );
            }
            if (json.contains(prefix + ".labelFormat") && json[prefix + ".labelFormat"].isString()) {
                axis.labelFormat = json[prefix + ".labelFormat"].toString();
                ui->lineEditFormat->setText( axis.labelFormat );
                ui->lineEditFormat->setCursorPosition(0);
            }
            if (json.contains(prefix + ".labelSize") && json[prefix + ".labelSize"].isDouble()) {
                axis.labelSize = json[prefix + ".labelSize"].toInt(8);
                ui->spinBoxLabelSize->setValue( axis.labelSize );
            }
            if (json.contains(prefix + ".ticks") && json[prefix + ".ticks"].isDouble()) {
                axis.ticks = json[prefix + ".ticks"].toInt(idx == 0 ? 9 : 5);
                ui->spinBoxTicks->setValue( axis.ticks );
            }
            if (json.contains(prefix + ".mticks") && json[prefix + ".mticks"].isDouble()) {
                axis.mticks = json[prefix + ".mticks"].toInt(0);
                ui->spinBoxMTicks->setValue( axis.mticks );
            }
            if (!init)
            {
                if (json.contains(prefix + ".titleText") && json[prefix + ".titleText"].isString()) {
                    axis.titleText = json[prefix + ".titleText"].toString();
                    ui->lineEditTitle->setText( axis.titleText );
                    ui->lineEditTitle->setCursorPosition(0);
                }
                if (idx == 1)
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

void PlotterLineChart::saveConfig()
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
        for (const auto& axis : mAxesParams)
        {
            json[prefix + ".visible"]     = axis.visible;
            json[prefix + ".title"]       = axis.title;
            json[prefix + ".log"]         = axis.log;
            json[prefix + ".logBase"]     = axis.logBase;
            json[prefix + ".titleText"]   = axis.titleText;
            json[prefix + ".titleSize"]   = axis.titleSize;
            json[prefix + ".labelFormat"] = axis.labelFormat;
            json[prefix + ".labelSize"]   = axis.labelSize;
            json[prefix + ".min"]         = axis.min;
            json[prefix + ".max"]         = axis.max;
            json[prefix + ".ticks"]       = axis.ticks;
            json[prefix + ".mticks"]      = axis.mticks;
            
            prefix = "axis.y";
        }
        
        configFile.write( QJsonDocument(json).toJson() );
    }
    else
        qWarning() << "Couldn't update: " << QString(config_folder) + config_file;
}

//
// Theme
void PlotterLineChart::onComboThemeChanged(int index)
{
    QChart::ChartTheme theme = static_cast<QChart::ChartTheme>(
                ui->comboBoxTheme->itemData(index).toInt());
    mChartView->chart()->setTheme(theme);
    
    // Update series color
    const auto& chartSeries = mChartView->chart()->series();
    for (int idx = 0 ; idx < mSeriesMapping.size(); ++idx)
    {
        auto& config = mSeriesMapping[idx];
        const auto& series = (QXYSeries*)chartSeries.at(idx);
        auto prevColor = config.oldColor;
        
        config.oldColor = series->color();
        if (config.newColor != prevColor)
            series->setColor(config.newColor); // re-apply config
        else
            config.newColor = config.oldColor; // sync with theme
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
void PlotterLineChart::onCheckLegendVisible(int state)
{
    mChartView->chart()->legend()->setVisible(state == Qt::Checked);
}

void PlotterLineChart::onComboLegendAlignChanged(int index)
{
    Qt::Alignment align = static_cast<Qt::Alignment>(
                ui->comboBoxLegendAlign->itemData(index).toInt());
    mChartView->chart()->legend()->setAlignment(align);
}

void PlotterLineChart::onSpinLegendFontSizeChanged(int i)
{
    QFont font = mChartView->chart()->legend()->font();
    font.setPointSize(i);
    mChartView->chart()->legend()->setFont(font);
}

void PlotterLineChart::onSeriesEditClicked()
{
    SeriesDialog seriesDialog(mSeriesMapping, this);
    auto res = seriesDialog.exec();
    if (res == QDialog::Accepted)
    {
        const auto& newMapping = seriesDialog.getMapping();
        for (int idx = 0; idx < newMapping.size(); ++idx)
        {
            const auto& newPair = newMapping[idx];
            const auto& oldPair = mSeriesMapping[idx];
            auto series = (QXYSeries*)mChartView->chart()->series().at(idx);
            if (newPair.newName != oldPair.newName) {
                series->setName( newPair.newName.toHtmlEscaped() );
            }
            if (newPair.newColor != oldPair.newColor) {
                series->setColor(newPair.newColor);
            }
        }
        mSeriesMapping = newMapping;
    }
}

void PlotterLineChart::onComboTimeUnitChanged(int /*index*/)
{
    if (mIgnoreEvents) return;
    
    // Update data
    double unitFactor = ui->comboBoxTimeUnit->currentData().toDouble();
    double updateFactor = unitFactor / mCurrentTimeFactor;  // can cause precision loss
    auto chartSeries = mChartView->chart()->series();
    for (auto& series : chartSeries)
    {
        auto xySeries = (QXYSeries*)series;
        auto points = xySeries->pointsVector();
        for (auto& point : points) {
            point.setY(point.y() * updateFactor);
        }
        xySeries->replace(points);
    }
    
    // Update axis title
    QString oldUnitName = "(us)";
    if      (mCurrentTimeFactor > 1.) oldUnitName = "(ns)";
    else if (mCurrentTimeFactor < 1.) oldUnitName = "(ms)";
    
    const auto& axes = mChartView->chart()->axes(Qt::Vertical);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        QString axisTitle = axis->titleText();
        if (axisTitle.endsWith(oldUnitName)) {
            QString unitName  = ui->comboBoxTimeUnit->currentText();
            onEditTitleChanged2(axisTitle.replace(axisTitle.size() - 3, 2, unitName), 1);
        }
    }
    // Update range
    if (ui->comboBoxAxis->currentIndex() == 1) {
        ui->doubleSpinBoxMin->setValue(mAxesParams[1].min * updateFactor);
        ui->doubleSpinBoxMax->setValue(mAxesParams[1].max * updateFactor);
    }
    else {
        onSpinMinChanged2(mAxesParams[1].min * updateFactor, 1);
        onSpinMaxChanged2(mAxesParams[1].max * updateFactor, 1);
    }
    
    mCurrentTimeFactor = unitFactor;
}

//
// Axes
void PlotterLineChart::onComboAxisChanged(int idx)
{
    // Update UI
    bool wasIgnoring = mIgnoreEvents;
    mIgnoreEvents = true;
    
    ui->checkBoxAxisVisible->setChecked( mAxesParams[idx].visible );
    ui->checkBoxTitle->setChecked( mAxesParams[idx].title );
    ui->checkBoxLog->setChecked( mAxesParams[idx].log );
    ui->spinBoxLogBase->setValue( mAxesParams[idx].logBase );
    ui->lineEditTitle->setText( mAxesParams[idx].titleText );
    ui->lineEditTitle->setCursorPosition(0);
    ui->spinBoxTitleSize->setValue( mAxesParams[idx].titleSize );
    ui->lineEditFormat->setText( mAxesParams[idx].labelFormat );
    ui->lineEditFormat->setCursorPosition(0);
    ui->spinBoxLabelSize->setValue( mAxesParams[idx].labelSize );
    ui->doubleSpinBoxMin->setDecimals(idx == 1 ? 6 : 3);
    ui->doubleSpinBoxMax->setDecimals(idx == 1 ? 6 : 3);
    ui->doubleSpinBoxMin->setValue( mAxesParams[idx].min );
    ui->doubleSpinBoxMax->setValue( mAxesParams[idx].max );
    ui->doubleSpinBoxMin->setSingleStep(idx == 1 ? 0.1 : 1.0);
    ui->doubleSpinBoxMax->setSingleStep(idx == 1 ? 0.1 : 1.0);
    ui->spinBoxTicks->setValue( mAxesParams[idx].ticks );
    ui->spinBoxMTicks->setValue( mAxesParams[idx].mticks );
    
    ui->spinBoxTicks->setEnabled( !mAxesParams[idx].log );
    ui->spinBoxLogBase->setEnabled( mAxesParams[idx].log );
    
    mIgnoreEvents = wasIgnoring;
}

void PlotterLineChart::onCheckAxisVisible(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setVisible(state == Qt::Checked);
        mAxesParams[iAxis].visible = state == Qt::Checked;
    }
}

void PlotterLineChart::onCheckTitleVisible(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setTitleVisible(state == Qt::Checked);
        mAxesParams[iAxis].title = state == Qt::Checked;
    }
}

void PlotterLineChart::onCheckLog(int state)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal  : Qt::Vertical;
    Qt::Alignment   align  = iAxis == 0 ? Qt::AlignBottom : Qt::AlignLeft;
    
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
            const auto chartSeries = mChartView->chart()->series();
            for (const auto& series : chartSeries)
                series->attachAxis(logAxis);
            
            logAxis->setBase( mAxesParams[iAxis].logBase );
            logAxis->setMin( mAxesParams[iAxis].min );
            logAxis->setMax( mAxesParams[iAxis].max );
            logAxis->setMinorTickCount( mAxesParams[iAxis].mticks );
        }
        else
        {
            QLogValueAxis*logAxis = (QLogValueAxis*)(axes.first());
    
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
            
            axis->setMin( mAxesParams[iAxis].min );
            axis->setMax( mAxesParams[iAxis].max );
            axis->setTickCount( mAxesParams[iAxis].ticks );
            axis->setMinorTickCount( mAxesParams[iAxis].mticks );
        }
        ui->spinBoxTicks->setEnabled(  state != Qt::Checked);
        ui->spinBoxLogBase->setEnabled(state == Qt::Checked);
        mAxesParams[iAxis].log = state == Qt::Checked;
    }
}

void PlotterLineChart::onSpinLogBaseChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() && ui->checkBoxLog->isChecked())
    {
        QLogValueAxis*logAxis = (QLogValueAxis*)(axes.first());
        logAxis->setBase(i);
        mAxesParams[iAxis].logBase = i;
    }
}

void PlotterLineChart::onEditTitleChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onEditTitleChanged2(text, iAxis);
}

void PlotterLineChart::onEditTitleChanged2(const QString& text, int iAxis)
{
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setTitleText(text);
        mAxesParams[iAxis].titleText = text;
    }
}

void PlotterLineChart::onSpinTitleSizeChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinTitleSizeChanged2(i, iAxis);
}

void PlotterLineChart::onSpinTitleSizeChanged2(int i, int iAxis)
{
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        
        QFont font = axis->titleFont();
        font.setPointSize(i);
        axis->setTitleFont(font);
        mAxesParams[iAxis].titleSize = i;
    }
}

void PlotterLineChart::onEditFormatChanged(const QString& text)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
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
        mAxesParams[iAxis].labelFormat = text;
    }
}

void PlotterLineChart::onSpinLabelSizeChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();

    onSpinLabelSizeChanged2(i, iAxis);
}

void PlotterLineChart::onSpinLabelSizeChanged2(int i, int iAxis)
{
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        
        QFont font = axis->labelsFont();
        font.setPointSize(i);
        axis->setLabelsFont(font);
        mAxesParams[iAxis].labelSize = i;
    }
}

void PlotterLineChart::onSpinMinChanged(double d)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinMinChanged2(d, iAxis);
}

void PlotterLineChart::onSpinMinChanged2(double d, int iAxis)
{
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setMin(d);
        mAxesParams[iAxis].min = d;
    }
}

void PlotterLineChart::onSpinMaxChanged(double d)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    
    onSpinMaxChanged2(d, iAxis);
}

void PlotterLineChart::onSpinMaxChanged2(double d, int iAxis)
{
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() ) {
        QAbstractAxis* axis = axes.first();
        axis->setMax(d);
        mAxesParams[iAxis].max = d;
    }
}

void PlotterLineChart::onSpinTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
    const auto& axes = mChartView->chart()->axes(orient);
    if ( !axes.isEmpty() )
    {
        if ( !ui->checkBoxLog->isChecked() ) {
            QValueAxis* axis = (QValueAxis*)(axes.first());
            axis->setTickCount(i);
            mAxesParams[iAxis].ticks = i;
        }
    }
}

void PlotterLineChart::onSpinMTicksChanged(int i)
{
    if (mIgnoreEvents) return;
    int iAxis = ui->comboBoxAxis->currentIndex();
    Qt::Orientation orient = iAxis == 0 ? Qt::Horizontal : Qt::Vertical;
    
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
        mAxesParams[iAxis].mticks = i;
    }
}

//
// Actions
void PlotterLineChart::onCheckAutoReload(int state)
{
    if (state == Qt::Checked)
    {
        if (mWatcher.files().empty())
        {
            mWatcher.addPath(mOrigFilename);
            for (const auto& addFilename : qAsConst(mAddFilenames))
                mWatcher.addPath( addFilename.filename );
        }
    }
    else
    {
        if (!mWatcher.files().empty())
            mWatcher.removePaths( mWatcher.files() );
    }
}

void PlotterLineChart::onAutoReload(const QString &path)
{
    QFileInfo fi(path);
    if (fi.exists() && fi.isReadable() && fi.size() > 0)
        onReloadClicked();
    else
        qWarning() << "Unable to auto-reload file: " << path;
}

void PlotterLineChart::onReloadClicked()
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
    const auto& oldChartSeries = mChartView->chart()->series();
    int newSeriesIdx = 0;
    if (errorMsg.isEmpty())
    {
        for (const auto& bchSubset : qAsConst(newBchSubsets))
        {
            // Ignore single point lines
            if (bchSubset.idxs.size() < 2)
                continue;
            if (newSeriesIdx >= oldChartSeries.size())
                break;
            
            const QString& subsetName = bchSubset.name;
            if (subsetName != mSeriesMapping[newSeriesIdx].oldName) {
                errorMsg = "Series has different name";
                break;
            }
            const auto lineSeries = (QLineSeries*)(oldChartSeries[newSeriesIdx]);
            if (bchSubset.idxs.size() != lineSeries->count())
            {
                errorMsg = "Series has different number of points";
                break;
            }
            ++newSeriesIdx;
        }
        if (newSeriesIdx != oldChartSeries.size()) {
            errorMsg = "Number of series is different";
        }
    }
    
    // Direct update if compatible
    if ( errorMsg.isEmpty() )
    {
        bool custDataAxis = true;
        QString custDataName;
        newSeriesIdx = 0;
        for (const auto& bchSubset : qAsConst(newBchSubsets))
        {
            // Ignore single point lines
            if (bchSubset.idxs.size() < 2) {
                qWarning() << "Not enough points to trace line for: " << bchSubset.name;
                continue;
            }
            
            // Update points
            QXYSeries* oldSeries = (QXYSeries*)oldChartSeries[newSeriesIdx];
            oldSeries->clear();
            
            double xFallback = 0.;
            for (int idx : bchSubset.idxs)
            {
                QString xName = newBchResults.getParamName(mPlotParams.xType == PlotArgumentType,
                                                           idx, mPlotParams.xIdx);
                double xVal = BenchResults::getParamValue(xName, custDataName, custDataAxis, xFallback);
                
                // Add point
                oldSeries->append(xVal, getYPlotValue(newBchResults.benchmarks[idx], mPlotParams.yType) * mCurrentTimeFactor);
            }
            ++newSeriesIdx;
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
    ui->labelLastReload->setText("(Last: " + now.toString() + ")");
}

void PlotterLineChart::onSnapshotClicked()
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
