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

#include "result_selector.h"
#include "ui_result_selector.h"

#include "result_parser.h"
#include "plot_parameters.h"

#include "plotter_linechart.h"
#include "plotter_barchart.h"
#include "plotter_boxchart.h"
#include "plotter_3dbars.h"
#include "plotter_3dsurface.h"

#include <QFileInfo>
#include <QFileDialog>
#include <QDateTime>
#include <QCollator>
#include <QMessageBox>
#include <QJsonObject>
#include <QJsonDocument>
#include <QScreen>
#include <QGuiApplication>

static const char* config_file = "config_selector.json";


ResultSelector::ResultSelector(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ResultSelector)
    , mWatcher(parent)
{
    ui->setupUi(this);
    
    this->setWindowTitle("JOMT");
    
    ui->pushButtonAppend->setEnabled(false);
    ui->pushButtonOverwrite->setEnabled(false);
    ui->pushButtonReload->setEnabled(false);
    ui->pushButtonSelectAll->setEnabled(false);
    ui->pushButtonSelectNone->setEnabled(false);
    ui->pushButtonPlot->setEnabled(false);
    
    connectUI();
    loadConfig();
}

ResultSelector::ResultSelector(const BenchResults &bchResults, const QString &fileName, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ResultSelector)
    , mBchResults(bchResults)
    , mWatcher(parent)
{
    ui->setupUi(this);
    
    if ( !fileName.isEmpty() ) {
        QFileInfo fileInfo(fileName);
        this->setWindowTitle("JOMT - " + fileInfo.fileName());
    }
    mOrigFilename = fileName; // For reload
    updateResults(false);
    
    connectUI();
    loadConfig();
    
    if (ui->checkBoxAutoReload->isChecked())
        onCheckAutoReload(Qt::Checked);
}

ResultSelector::~ResultSelector()
{
    saveConfig();
    delete ui;
}

// Private
void ResultSelector::connectUI()
{
    connect(ui->treeWidget, &QTreeWidget::itemChanged, this, &ResultSelector::onItemChanged);
    
    connect(ui->comboBoxType, QOverload<int>::of(&QComboBox::activated), this, &ResultSelector::onComboTypeChanged);
    connect(ui->comboBoxX,    QOverload<int>::of(&QComboBox::activated), this, &ResultSelector::onComboXChanged);
    connect(ui->comboBoxZ,    QOverload<int>::of(&QComboBox::activated), this, &ResultSelector::onComboZChanged);
    
    connect(&mWatcher,              &QFileSystemWatcher::fileChanged, this, &ResultSelector::onAutoReload);
    connect(ui->checkBoxAutoReload, &QCheckBox::stateChanged,         this, &ResultSelector::onCheckAutoReload);
    connect(ui->pushButtonReload,   &QPushButton::clicked,            this, &ResultSelector::onReloadClicked);
    
    connect(ui->pushButtonNew,       &QPushButton::clicked, this, &ResultSelector::onNewClicked);
    connect(ui->pushButtonAppend,    &QPushButton::clicked, this, &ResultSelector::onAppendClicked);
    connect(ui->pushButtonOverwrite, &QPushButton::clicked, this, &ResultSelector::onOverwriteClicked);
    
    connect(ui->pushButtonSelectAll,  &QPushButton::clicked, this, &ResultSelector::onSelectAllClicked);
    connect(ui->pushButtonSelectNone, &QPushButton::clicked, this, &ResultSelector::onSelectNoneClicked);
    
    connect(ui->pushButtonPlot, &QPushButton::clicked, this, &ResultSelector::onPlotClicked);
}

void ResultSelector::loadConfig()
{
    // Load options from file
    QFile configFile(QString(config_folder) + config_file);
    if (configFile.open(QIODevice::ReadOnly))
    {
        QByteArray configData = configFile.readAll();
        configFile.close();
        QJsonDocument configDoc(QJsonDocument::fromJson(configData));
        QJsonObject json = configDoc.object();
        
        // FileDialog
        if (json.contains("workingDir") && json["workingDir"].isString())
            mWorkingDir = json["workingDir"].toString();
        // Actions
        if (json.contains("autoReload") && json["autoReload"].isBool())
            ui->checkBoxAutoReload->setChecked( json["autoReload"].toBool() );
    }
    else
    {
        if (configFile.exists())
            qWarning() << "Couldn't read: " << QString(config_folder) + config_file;
    }
    
    // Default size
    QSize size = this->size();
    QSize newSize = QGuiApplication::primaryScreen()->size();
    newSize *= 0.375f;
    if (newSize.width() > size.width() && newSize.height() > size.height())
        this->resize(newSize.height() * 16.f/9.f, newSize.height());
}

void ResultSelector::saveConfig()
{
    // Save options to file
    QFile configFile(QString(config_folder) + config_file);
    if (configFile.open(QIODevice::WriteOnly))
    {
        QJsonObject json;
        
        // FileDialog
        json["workingDir"] = mWorkingDir;
        // Actions
        json["autoReload"] = ui->checkBoxAutoReload->isChecked();
        
        configFile.write( QJsonDocument(json).toJson() );
    }
    else
        qWarning() << "Couldn't update: " << QString(config_folder) + config_file;
}

void ResultSelector::updateComboBoxY()
{
    PlotChartType chartType = (PlotChartType)ui->comboBoxType->currentData().toInt();
    PlotValueType prevYType = (PlotValueType)-1;
    if (ui->comboBoxY->count() > 0)
        prevYType = (PlotValueType)ui->comboBoxY->currentData().toInt();
    
    // Classic or Boxes
    if (!mBchResults.meta.hasAggregate || chartType == ChartBoxType)
    {
        ui->comboBoxY->clear();
        
        ui->comboBoxY->addItem("Real time",     QVariant(RealTimeType));
        ui->comboBoxY->addItem("CPU time",      QVariant(CpuTimeType));
        ui->comboBoxY->addItem("Iterations",    QVariant(IterationsType));
        if (mBchResults.meta.hasBytesSec)
            ui->comboBoxY->addItem("Bytes/s",   QVariant(BytesType));
        if (mBchResults.meta.hasItemsSec)
            ui->comboBoxY->addItem("Items/s",   QVariant(ItemsType));
    }
    // Aggregate
    else
    {
        ui->comboBoxY->clear();
        
        if (!mBchResults.meta.onlyAggregate)
            ui->comboBoxY->addItem("Real min time",     QVariant(RealTimeMinType));
        ui->comboBoxY->addItem("Real mean time",        QVariant(RealTimeMeanType));
        ui->comboBoxY->addItem("Real median time",      QVariant(RealTimeMedianType));
        ui->comboBoxY->addItem("Real stddev time",      QVariant(RealTimeStddevType));
        if (mBchResults.meta.hasCv)
            ui->comboBoxY->addItem("Real cv percent",   QVariant(RealTimeCvType));
        
        if (!mBchResults.meta.onlyAggregate)
            ui->comboBoxY->addItem("CPU min time",      QVariant(CpuTimeMinType));
        ui->comboBoxY->addItem("CPU mean time",         QVariant(CpuTimeMeanType));
        ui->comboBoxY->addItem("CPU median time",       QVariant(CpuTimeMedianType));
        ui->comboBoxY->addItem("CPU stddev time",       QVariant(CpuTimeStddevType));
        if (mBchResults.meta.hasCv)
            ui->comboBoxY->addItem("CPU cv percent",    QVariant(CpuTimeCvType));
        
        ui->comboBoxY->addItem("Iterations",            QVariant(IterationsType));
        
        if (mBchResults.meta.hasBytesSec) {
            ui->comboBoxY->addItem("Bytes/s min",       QVariant(BytesMinType));
            ui->comboBoxY->addItem("Bytes/s mean",      QVariant(BytesMeanType));
            ui->comboBoxY->addItem("Bytes/s median",    QVariant(BytesMedianType));
            ui->comboBoxY->addItem("Bytes/s stddev",    QVariant(BytesStddevType));
            if (mBchResults.meta.hasCv)
                ui->comboBoxY->addItem("Bytes/s cv",    QVariant(BytesCvType));
        }
        if (mBchResults.meta.hasItemsSec) {
            ui->comboBoxY->addItem("Items/s min",       QVariant(ItemsMinType));
            ui->comboBoxY->addItem("Items/s mean",      QVariant(ItemsMeanType));
            ui->comboBoxY->addItem("Items/s median",    QVariant(ItemsMedianType));
            ui->comboBoxY->addItem("Items/s stddev",    QVariant(ItemsStddevType));
            if (mBchResults.meta.hasCv)
                ui->comboBoxY->addItem("Items/s cv",    QVariant(ItemsCvType));
        }
    }
    // Restore
    int yIdx = ui->comboBoxY->findData(prevYType);
    if (yIdx >= 0)
        ui->comboBoxY->setCurrentIndex(yIdx);
}

// Tree
class NumericTreeWidgetItem : public QTreeWidgetItem
{
public:
  NumericTreeWidgetItem(const QStringList &strings)
      : QTreeWidgetItem(strings) {}
private:
  bool operator<(const QTreeWidgetItem &other) const
  {
      QCollator collator;
      collator.setNumericMode(true);
      int column = treeWidget()->sortColumn();
      return collator.compare( text( column ), other.text( column ) ) < 0;
  }
};

static QTreeWidgetItem* buildTreeItem(const BenchData &bchData, double timeFactor, bool onlyAggregate, QTreeWidgetItem *item = nullptr)
{
    QStringList labels = {
        bchData.base_name, bchData.templates.join(", "), bchData.arguments.join("/"),
        QString::number((!onlyAggregate ? bchData.real_time_us : bchData.mean_real) * timeFactor),
        QString::number((!onlyAggregate ? bchData.cpu_time_us  : bchData.mean_cpu)  * timeFactor)
    };
    if ( !bchData.kbytes_sec.isEmpty() )
        labels.append( QString::number(bchData.kbytes_sec_dflt) );
    if ( !bchData.kitems_sec.isEmpty() )
        labels.append( QString::number(bchData.kitems_sec_dflt) );
    
    if (item == nullptr) {
        item = new NumericTreeWidgetItem(labels);
    }
    else {
        for (int iC=0; iC<labels.size(); ++iC)
            item->setText(iC, labels[iC]);
    }
    
    return item;
}

static QSet<QString> getUnselectedBenchmarks(const QTreeWidget *tree, const BenchResults &bchResults)
{
    QSet<QString> resNames;
    
    for (int i=0; i<tree->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem *topItem = tree->topLevelItem(i);
        if (topItem->childCount() <= 0)
        {
            if (topItem->checkState(0) == Qt::Unchecked) {
                int idx = topItem->data(0, Qt::UserRole).toInt();
                resNames.insert( bchResults.getBenchName(idx) );
            }
        }
        else
        {
            for (int j=0; j<topItem->childCount(); ++j)
            {
                QTreeWidgetItem *midItem = topItem->child(j);
                if (midItem->childCount() <= 0)
                {
                    if (midItem->checkState(0) == Qt::Unchecked) {
                        int idx = midItem->data(0, Qt::UserRole).toInt();
                        resNames.insert( bchResults.getBenchName(idx) );
                    }
                }
                else
                {
                    for (int k=0; k<midItem->childCount(); ++k)
                    {
                        QTreeWidgetItem *lowItem = midItem->child(k);
                        if (lowItem->checkState(0) == Qt::Unchecked) {
                            int idx = lowItem->data(0, Qt::UserRole).toInt();
                            resNames.insert( bchResults.getBenchName(idx) );
                        }
                    }
                }
            }
        }
    }
    
    return resNames;
}

void ResultSelector::updateResults(bool clear, const QSet<QString> unselected)
{
    //
    // Tree widget
//    QSet<QString> unselected;
    if (clear)
    {
//        if (keepSelection)
//            unselected = getUnselectedBenchmarks(ui->treeWidget, mBchResults);
        ui->treeWidget->clear();
    }
    else
        ui->treeWidget->sortByColumn(-1, Qt::SortOrder::AscendingOrder); // init: unsorted
    ui->treeWidget->setSortingEnabled(true);
    
    // Columns
    int iCol = 5;
    if (mBchResults.meta.hasBytesSec) ++iCol;
    if (mBchResults.meta.hasItemsSec) ++iCol;
    ui->treeWidget->setColumnCount(iCol);
    
    // Time unit
    double timeFactor = 1.;
    if (     mBchResults.meta.time_unit == "ns") timeFactor = 1000.;
    else if (mBchResults.meta.time_unit == "ms") timeFactor = 0.001;
    else                                         mBchResults.meta.time_unit = "us";
    
    // Populate tree
    bool anySelected = false;
    QList<QTreeWidgetItem *> items;
    
    QVector<BenchSubset> bchFamilies = mBchResults.segmentFamilies();
    for (const auto &bchFamily : qAsConst(bchFamilies))
    {
        bool oneTopSelected = false;
        bool allTopSelected = true;
        QTreeWidgetItem* topItem = new QTreeWidgetItem( QStringList(bchFamily.name) );
        
        // JOMT: family + container
        if ( !mBchResults.benchmarks[bchFamily.idxs[0]].container.isEmpty() )
        {
            QVector<BenchSubset> bchContainers = mBchResults.segmentContainers(bchFamily.idxs);
            for (const auto &bchContainer : qAsConst(bchContainers))
            {
                bool oneMidSelected = false;
                bool allMidSelected = true;
                QTreeWidgetItem* midItem = new QTreeWidgetItem( QStringList(bchContainer.name) );
                
                for (int idx : bchContainer.idxs)
                {
                    QTreeWidgetItem *child = buildTreeItem(mBchResults.benchmarks[idx], timeFactor, mBchResults.meta.onlyAggregate);
                    bool selected = !unselected.contains( mBchResults.getBenchName(idx) );
                    oneMidSelected |= selected;
                    allMidSelected &= selected;
                    child->setCheckState(0, selected ? Qt::Checked : Qt::Unchecked);
                    child->setData(0, Qt::UserRole, idx);
                    midItem->addChild(child);
                }
                oneTopSelected |= oneMidSelected;
                allTopSelected &= allMidSelected;
                midItem->setCheckState(0, oneMidSelected ? (allMidSelected ? Qt::Checked : Qt::PartiallyChecked) : Qt::Unchecked);
                topItem->addChild(midItem);
            }
        }
        // Classic
        else
        {
            // Single
            if (bchFamily.idxs.size() == 1)
            {
                int idx = bchFamily.idxs[0];
                buildTreeItem(mBchResults.benchmarks[idx], timeFactor, mBchResults.meta.onlyAggregate, topItem);
                oneTopSelected = !unselected.contains( mBchResults.getBenchName(idx) );
                topItem->setData(0, Qt::UserRole, idx);
            }
            else // Family
            {
                for (int idx : bchFamily.idxs)
                {
                    QTreeWidgetItem *child = buildTreeItem(mBchResults.benchmarks[idx], timeFactor, mBchResults.meta.onlyAggregate);
                    bool selected = !unselected.contains( mBchResults.getBenchName(idx) );
                    oneTopSelected |= selected;
                    allTopSelected &= selected;
                    child->setCheckState(0, selected ? Qt::Checked : Qt::Unchecked);
                    child->setData(0, Qt::UserRole, idx);
                    topItem->addChild(child);
                }
            }
        }
        anySelected |= oneTopSelected;
        topItem->setCheckState(0, oneTopSelected ? (allTopSelected ? Qt::Checked : Qt::PartiallyChecked) : Qt::Unchecked);
        items.insert(0, topItem);
    }
    ui->treeWidget->insertTopLevelItems(0, items);
    
    ui->pushButtonPlot->setEnabled(anySelected);
    
    // Headers
    QStringList labels = {"Benchmark", "Templates", "Arguments"};
    if (!mBchResults.meta.hasAggregate) {
        labels << "Real time (" + mBchResults.meta.time_unit + ")"
               << "CPU time ("  + mBchResults.meta.time_unit + ")";
        if (mBchResults.meta.hasBytesSec) labels << "Bytes/s (k)";
        if (mBchResults.meta.hasItemsSec) labels << "Items/s (k)";
    }
    else {
        if (!mBchResults.meta.onlyAggregate) {
            labels << "Real min time (" + mBchResults.meta.time_unit + ")"
                   << "CPU min time ("  + mBchResults.meta.time_unit + ")";
        }
        else {
            labels << "Real mean time (" + mBchResults.meta.time_unit + ")"
                   << "CPU mean time ("  + mBchResults.meta.time_unit + ")";
        }
        if (mBchResults.meta.hasBytesSec) labels << "Bytes/s min (k)";
        if (mBchResults.meta.hasItemsSec) labels << "Items/s min (k)";
    }
    
    ui->treeWidget->setHeaderLabels(labels);
    
    ui->treeWidget->expandAll();
    for (int iC=0; iC<ui->treeWidget->columnCount(); ++iC)
        ui->treeWidget->resizeColumnToContents(iC);
    
    
    //
    // Chart options
    PlotChartType prevChartType = (PlotChartType)-1;
    PlotParamType prevXType = PlotEmptyType;
    PlotValueType prevYType = (PlotValueType)-1;
    PlotParamType prevZType = PlotEmptyType;
    if (clear)
    {
        if (ui->comboBoxType->count() > 0)
        {
            prevChartType = (PlotChartType)ui->comboBoxType->currentData().toInt();
            if (ui->comboBoxX->count() > 0)
                prevXType = (PlotParamType)ui->comboBoxX->currentData().toList()[0].toInt();
            if (ui->comboBoxY->count() > 0)
                prevYType = (PlotValueType)ui->comboBoxY->currentData().toInt();
            if (ui->comboBoxZ->count() > 0)
                prevZType = (PlotParamType)ui->comboBoxZ->currentData().toList()[0].toInt();
        }
        ui->comboBoxType->clear();
        ui->comboBoxX->clear();
        ui->comboBoxY->clear();
        ui->comboBoxZ->clear();
    }
    
    // Type
    if (mBchResults.meta.maxArguments > 0 || mBchResults.meta.maxTemplates > 0) {
        ui->comboBoxType->addItem("Lines",      ChartLineType);
        ui->comboBoxType->addItem("Splines",    ChartSplineType);
    }
    ui->comboBoxType->addItem("Bars",   ChartBarType);
    ui->comboBoxType->addItem("HBars",  ChartHBarType);
    if (mBchResults.meta.hasAggregate && !mBchResults.meta.onlyAggregate)
        ui->comboBoxType->addItem("Boxes",  ChartBoxType);
    ui->comboBoxType->addItem("3D Bars",    Chart3DBarsType);
    if (mBchResults.meta.maxArguments > 0 || mBchResults.meta.maxTemplates > 0)
        ui->comboBoxType->addItem("3D Surface", Chart3DSurfaceType);
    
    
    // X-axis
    for (int i=0; i<mBchResults.meta.maxArguments; ++i)
    {
        QList<QVariant> qvList;
        qvList.append(PlotArgumentType); qvList.append(i);
        ui->comboBoxX->addItem("Argument " + QString::number(i+1), qvList);
    }
    for (int i=0; i<mBchResults.meta.maxTemplates; ++i)
    {
        QList<QVariant> qvList;
        qvList.append(PlotTemplateType); qvList.append(i);
        ui->comboBoxX->addItem("Template " + QString::number(i+1), qvList);
    }
    ui->comboBoxX->setEnabled(ui->comboBoxX->count() > 0);
    
    // Y-axis
    updateComboBoxY();
    
    // Z-axis
    if ( !ui->comboBoxX->isEnabled() ) {
        ui->comboBoxZ->setEnabled(false);
    }
    else
    {
        {
            QList<QVariant> qvList;
            qvList.append(PlotEmptyType); qvList.append(0);
            ui->comboBoxZ->addItem("Auto", qvList);
        }
        
        PlotChartType chartType = (PlotChartType)ui->comboBoxType->currentData().toInt();
        if (chartType == Chart3DBarsType || chartType == Chart3DSurfaceType) // Any 3D charts
            ui->comboBoxZ->setEnabled(true);
        else
            ui->comboBoxZ->setEnabled(false);

        for (int i=0; i<mBchResults.meta.maxArguments; ++i)
        {
            QList<QVariant> qvList;
            qvList.append(PlotArgumentType); qvList.append(i);
            ui->comboBoxZ->addItem("Argument " + QString::number(i+1), qvList);
        }
        for (int i=0; i<mBchResults.meta.maxTemplates; ++i)
        {
            QList<QVariant> qvList;
            qvList.append(PlotTemplateType); qvList.append(i);
            ui->comboBoxZ->addItem("Template " + QString::number(i+1), qvList);
        }
    }
    
    // Restore options (if possible)
    if (prevChartType >= 0)
    {
        int chartIdx = ui->comboBoxType->findData(prevChartType);
        if (chartIdx >= 0) {
            ui->comboBoxType->setCurrentIndex(chartIdx);
            updateComboBoxY();
        }
        // X
        if (prevXType >= 0) {
            for (int i=0; i<ui->comboBoxX->count(); ++i) {
                if ((PlotParamType)ui->comboBoxX->itemData(i).toList()[0].toInt() == prevXType)
                    ui->comboBoxX->setCurrentIndex(i);
            }
        }
        // Y
        int yIdx = ui->comboBoxY->findData(prevYType);
        if (yIdx >= 0)
            ui->comboBoxY->setCurrentIndex(yIdx);
        // Z
        if (prevZType >= 0) {
            for (int i=0; i<ui->comboBoxZ->count(); ++i) {
                if ((PlotParamType)ui->comboBoxZ->itemData(i).toList()[0].toInt() == prevZType)
                    ui->comboBoxZ->setCurrentIndex(i);
            }
        }
        ui->comboBoxZ->setEnabled(ui->comboBoxX->isEnabled()
            && (prevChartType == Chart3DBarsType || prevChartType == Chart3DSurfaceType)); // Any 3D charts
    }
    
    // Reload
    ui->checkBoxAutoReload->setEnabled(true);
    ui->labelLastReload->setEnabled(true);
    ui->pushButtonReload->setEnabled(true);
    
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() + ")");
}

// Slots
static void updateItemParentsState(QTreeWidgetItem *item)
{
    auto parent = item->parent();
    if (parent == nullptr) return;
    
    bool allChecked = true, allUnchecked = true;
    for (int idx=0; (allChecked || allUnchecked) && idx<parent->childCount(); ++idx) {
        allChecked   &= parent->child(idx)->checkState(0) == Qt::Checked;
        allUnchecked &= parent->child(idx)->checkState(0) == Qt::Unchecked;
    }
    
    if      (allChecked)    parent->setCheckState(0, Qt::Checked);
    else if (allUnchecked)  parent->setCheckState(0, Qt::Unchecked);
    else                    parent->setCheckState(0, Qt::PartiallyChecked);
}

static void updateItemChildrenState(QTreeWidgetItem *item)
{
    if (item->childCount() <= 0) return;
    
    if (item->checkState(0) == Qt::Checked) {
        for (int idx=0; idx<item->childCount(); ++idx) {
            item->child(idx)->setCheckState(0, Qt::Checked);
        }
    }
    else if (item->checkState(0) == Qt::Unchecked)
    {
        for (int idx=0; idx<item->childCount(); ++idx) {
            item->child(idx)->setCheckState(0, Qt::Unchecked);
        }
    }
    // Nothing if 'PartiallyChecked'
}

void ResultSelector::onItemChanged(QTreeWidgetItem *item, int /*column*/)
{
    if (item == nullptr) return;
    
    updateItemChildrenState(item);
    updateItemParentsState(item);
    
    // Disable plot button if no items selected
    bool allUnchecked = true;
    for (int i=0; allUnchecked && i<ui->treeWidget->topLevelItemCount(); ++i)
        allUnchecked &= ui->treeWidget->topLevelItem(i)->checkState(0) == Qt::Unchecked;
    
    ui->pushButtonPlot->setEnabled(!allUnchecked);
}

static QVector<int> getSelectedBenchmarks(const QTreeWidget *tree)
{
    QVector<int> resIdxs;
    
    for (int i=0; i<tree->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem *topItem = tree->topLevelItem(i);
        if (topItem->childCount() <= 0)
        {
            if (topItem->checkState(0) == Qt::Checked)
                resIdxs.append( topItem->data(0, Qt::UserRole).toInt() );
        }
        else
        {
            for (int j=0; j<topItem->childCount(); ++j)
            {
                QTreeWidgetItem *midItem = topItem->child(j);
                if (midItem->childCount() <= 0)
                {
                    if (midItem->checkState(0) == Qt::Checked)
                        resIdxs.append( midItem->data(0, Qt::UserRole).toInt() );
                }
                else
                {
                    for (int k=0; k<midItem->childCount(); ++k)
                    {
                        QTreeWidgetItem *lowItem = midItem->child(k);
                        if (lowItem->checkState(0) == Qt::Checked)
                            resIdxs.append( lowItem->data(0, Qt::UserRole).toInt() );
                    }
                }
            }
        }
    }
    
    return resIdxs;
}


void ResultSelector::onComboTypeChanged(int /*index*/)
{
    PlotChartType chartType = (PlotChartType)ui->comboBoxType->currentData().toInt();
    
    if (chartType == Chart3DBarsType || chartType == Chart3DSurfaceType) // Any 3D charts
        ui->comboBoxZ->setEnabled( ui->comboBoxX->isEnabled() );
    else
        ui->comboBoxZ->setEnabled(false);
    
    if (mBchResults.meta.hasAggregate)
        updateComboBoxY();
}

void ResultSelector::onComboXChanged(int /*index*/)
{
    PlotParamType xType = (PlotParamType)ui->comboBoxX->currentData().toList()[0].toInt();
    int xIdx  = ui->comboBoxX->currentData().toList()[1].toInt();
    
    PlotParamType zType = (PlotParamType)ui->comboBoxZ->currentData().toList()[0].toInt();
    int zIdx  = ui->comboBoxZ->currentData().toList()[1].toInt();
    
    // Change comboZ to avoid having same value
    if (xType == zType && xIdx == zIdx)
    {
        if (ui->comboBoxZ->currentIndex() == 0)
            ui->comboBoxZ->setCurrentIndex(1);
        else
            ui->comboBoxZ->setCurrentIndex(0);
    }
}

void ResultSelector::onComboZChanged(int /*index*/)
{
    PlotParamType xType = (PlotParamType)ui->comboBoxX->currentData().toList()[0].toInt();
    int xIdx  = ui->comboBoxX->currentData().toList()[1].toInt();
    
    PlotParamType zType = (PlotParamType)ui->comboBoxZ->currentData().toList()[0].toInt();
    int zIdx  = ui->comboBoxZ->currentData().toList()[1].toInt();
    
    // Change comboX to avoid having same value
    if (zType == xType && zIdx == xIdx)
    {
        if (ui->comboBoxX->currentIndex() == 0)
            ui->comboBoxX->setCurrentIndex(1);
        else
            ui->comboBoxX->setCurrentIndex(0);
    }
}

// Reload
void ResultSelector::onAutoReload(const QString &path)
{
    QFileInfo fi(path);
    if (fi.exists() && fi.isReadable() && fi.size() > 0)
        onReloadClicked();
    else
        qWarning() << "Unable to auto-reload file: " << path;
}

void ResultSelector::updateReloadWatchList()
{
    if (ui->checkBoxAutoReload->isChecked())
    {
        if (!mWatcher.files().empty())
            mWatcher.removePaths( mWatcher.files() );
        
        mWatcher.addPath(mOrigFilename);
        for (const auto& addFilename : qAsConst(mAddFilenames))
            mWatcher.addPath( addFilename.filename );
    }
}

void ResultSelector::onCheckAutoReload(int state)
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

void ResultSelector::onReloadClicked()
{
    // Check original
    if ( mOrigFilename.isEmpty() ) {
        QMessageBox::warning(this, "Reload benchmark results", "No file to reload");
        return;
    }
    if ( !QFile::exists(mOrigFilename) ) {
        QMessageBox::warning(this, "Reload benchmark results",
                             "File to reload does no exist:" + mOrigFilename);
        return;
    }
    // Load original
    QString errorMsg;
    BenchResults newResults = ResultParser::parseJsonFile(mOrigFilename, errorMsg);
    if (newResults.benchmarks.size() <= 0) {
        QMessageBox::warning(this, "Reload benchmark results",
                             "Error parsing file: " + mOrigFilename + "\n" + errorMsg);
        return;
    }
    
    // Load additionnals
    for (const auto &addFile : qAsConst(mAddFilenames))
    {
        QString errorMsg;
        BenchResults addResults = ResultParser::parseJsonFile(addFile.filename, errorMsg);
        if (addResults.benchmarks.size() <= 0) {
            QMessageBox::warning(this, "Reload benchmark results",
                                 "Error parsing file: " + addFile.filename + "\n" + errorMsg);
            return;
        }
        // Append / Overwrite
        if (addFile.isAppend)
            newResults.appendResults(addResults);
        else
            newResults.overwriteResults(addResults);
    }
    
    // Replace & update
    auto unselected = getUnselectedBenchmarks(ui->treeWidget, mBchResults);
    mBchResults = newResults;
    updateResults(true, unselected);
    
    // Update timestamp
    QDateTime today = QDateTime::currentDateTime();
    QTime now = today.time();
    ui->labelLastReload->setText("(Last: " + now.toString() + ")");
}

// File
void ResultSelector::onNewClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open benchmark results"), mWorkingDir, tr("Benchmark results (*.json)"));
    
    if ( !fileName.isEmpty() && QFile::exists(fileName) )
    {
        QString errorMsg;
        BenchResults newResults = ResultParser::parseJsonFile(fileName, errorMsg);
        if (newResults.benchmarks.size() <= 0) {
            QMessageBox::warning(this, "Open benchmark results",
                                 "Error parsing file: " + fileName + "\n" + errorMsg);
            return;
        }
        // Replace & upate
        mBchResults = newResults;
        ui->treeWidget->sortByColumn(-1, Qt::SortOrder::AscendingOrder); // reset sorting
        updateResults(true);
        
        // Update UI
        ui->pushButtonAppend->setEnabled(true);
        ui->pushButtonOverwrite->setEnabled(true);
        ui->pushButtonReload->setEnabled(true);
        ui->pushButtonSelectAll->setEnabled(true);
        ui->pushButtonSelectNone->setEnabled(true);
        ui->pushButtonPlot->setEnabled(true);
        
        // Save for reload
        mOrigFilename = fileName;
        mAddFilenames.clear();
        updateReloadWatchList();
        
        // Window title
        QFileInfo fileInfo(fileName);
        this->setWindowTitle("JOMT - " +  fileInfo.fileName());
        
        mWorkingDir = fileInfo.absoluteDir().absolutePath();
    }
}

void ResultSelector::onAppendClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Append benchmark results"), mWorkingDir, tr("Benchmark results (*.json)"));
    
    if ( !fileName.isEmpty() && QFile::exists(fileName) )
    {
        QString errorMsg;
        BenchResults newResults = ResultParser::parseJsonFile(fileName, errorMsg);
        if (newResults.benchmarks.size() <= 0) {
            QMessageBox::warning(this, "Open benchmark results",
                                 "Error parsing file: " + fileName + "\n" + errorMsg);
            return;
        }
        // Append & upate
        auto unselected = getUnselectedBenchmarks(ui->treeWidget, mBchResults);
        mBchResults.appendResults(newResults);
        updateResults(true, unselected);
        
        // Save for reload
        mAddFilenames.append( {fileName, true} );
        updateReloadWatchList();
        
        // Window title
        if ( !this->windowTitle().endsWith(" + ...") )
            this->setWindowTitle( this->windowTitle() + " + ..." );
        
        QFileInfo fileInfo(fileName);
        mWorkingDir = fileInfo.absoluteDir().absolutePath();
    }
}

void ResultSelector::onOverwriteClicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Overwrite benchmark results"), mWorkingDir, tr("Benchmark results (*.json)"));
    
    if ( !fileName.isEmpty() && QFile::exists(fileName) )
    {
        QString errorMsg;
        BenchResults newResults = ResultParser::parseJsonFile(fileName, errorMsg);
        if (newResults.benchmarks.size() <= 0) {
            QMessageBox::warning(this, "Open benchmark results",
                                 "Error parsing file: " + fileName + "\n" + errorMsg);
            return;
        }
        // Overwrite & upate
        auto unselected = getUnselectedBenchmarks(ui->treeWidget, mBchResults);
        mBchResults.overwriteResults(newResults);
        updateResults(true, unselected);
        
        // Save for reload
        mAddFilenames.append( {fileName, false} );
        updateReloadWatchList();
        
        // Window title
        if ( !this->windowTitle().endsWith(" + ...") )
            this->setWindowTitle( this->windowTitle() + " + ..." );
        
        QFileInfo fileInfo(fileName);
        mWorkingDir = fileInfo.absoluteDir().absolutePath();
    }
}

// Selection
void ResultSelector::onSelectAllClicked()
{
    for (int i=0; i<ui->treeWidget->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem *topItem = ui->treeWidget->topLevelItem(i);
        topItem->setCheckState(0, Qt::Checked);
    }
}

void ResultSelector::onSelectNoneClicked()
{
    for (int i=0; i<ui->treeWidget->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem *topItem = ui->treeWidget->topLevelItem(i);
        topItem->setCheckState(0, Qt::Unchecked);
    }
}

// Plot
void ResultSelector::onPlotClicked()
{
    // Params
    PlotParams plotParams;
    
    plotParams.type = (PlotChartType)ui->comboBoxType->currentData().toInt();
    
    // Axes
    if (ui->comboBoxX->currentIndex() >= 0) {
        plotParams.xType = (PlotParamType)ui->comboBoxX->currentData().toList()[0].toInt();
        plotParams.xIdx  = ui->comboBoxX->currentData().toList()[1].toInt();
    }
    else {
        plotParams.xType = PlotEmptyType;
        plotParams.xIdx  = -1;
    }
    
    plotParams.yType = (PlotValueType)ui->comboBoxY->currentData().toInt();
    
    if ( ui->comboBoxZ->isEnabled() && ui->comboBoxZ->currentIndex() >= 0) {
        plotParams.zType = (PlotParamType)ui->comboBoxZ->currentData().toList()[0].toInt();
        plotParams.zIdx  = ui->comboBoxZ->currentData().toList()[1].toInt();
    }
    else {
        plotParams.zType = PlotEmptyType;
        plotParams.zIdx  = -1;
    }
    
    // Selected items
    const auto &bchIdxs = getSelectedBenchmarks(ui->treeWidget);

    //
    // Call plotter
    bool is3D = false;
    QWidget* widget = nullptr;
    switch (plotParams.type)
    {
        case ChartLineType:
        case ChartSplineType:
        {
            widget = new PlotterLineChart(mBchResults, bchIdxs,
                                          plotParams, mOrigFilename, mAddFilenames);
            break;
        }
        case ChartBarType:
        case ChartHBarType:
        {
            widget = new PlotterBarChart(mBchResults, bchIdxs,
                                         plotParams, mOrigFilename, mAddFilenames);
            break;
        }
        case ChartBoxType:
        {
            widget = new PlotterBoxChart(mBchResults, bchIdxs,
                                         plotParams, mOrigFilename, mAddFilenames);
            break;
        }
        case Chart3DBarsType:
        {
            widget = new Plotter3DBars(mBchResults, bchIdxs,
                                       plotParams, mOrigFilename, mAddFilenames);
            is3D = true;
            break;
        }
        case Chart3DSurfaceType:
        {
            widget = new Plotter3DSurface(mBchResults, bchIdxs,
                                          plotParams, mOrigFilename, mAddFilenames);
            is3D = true;
            break;
        }
    }
    
    if (widget)
    {
        // Default size
        QSize newSize = widget->size();
        QSize screenSize =  QGuiApplication::primaryScreen()->size();
        float scale = screenSize.height() * 0.375f / newSize.height();
        float ratio = 16.f/9.f;
        float h3DScale = 1.15f;
        if (scale > 1.f) {
            newSize *= scale;
            ratio = 2.3f;
            h3DScale = 1.5f;
        }
        newSize.setWidth(newSize.height() * ratio);
        if (is3D)
            newSize.setHeight(newSize.height() * h3DScale);
        widget->resize(newSize.width(), newSize.height());
        
        widget->show();
    }
    else
        qWarning() << "Unable to instantiate plot widget";
}
