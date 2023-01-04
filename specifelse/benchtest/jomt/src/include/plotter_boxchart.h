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

#ifndef PLOTTER_BOXCHART_H
#define PLOTTER_BOXCHART_H

#include "plot_parameters.h"
#include "series_dialog.h"

#include <QWidget>
#include <QVector>
#include <QString>
#include <QFileSystemWatcher>

namespace Ui {
class PlotterBoxChart;
}
namespace QtCharts {
class QChartView;
}
struct BenchResults;
struct FileReload;


class PlotterBoxChart : public QWidget
{
    Q_OBJECT
    
public:
    explicit PlotterBoxChart(BenchResults &bchResults, const QVector<int> &bchIdxs,
                             const PlotParams &plotParams, const QString &filename,
                             const QVector<FileReload>& addFilenames, QWidget *parent = nullptr);
    ~PlotterBoxChart();

private:
    void connectUI();
    void setupChart(BenchResults &bchResults, const QVector<int> &bchIdxs, const PlotParams &plotParams, bool init = true);
    void setupOptions(bool init = true);
    void loadConfig(bool init);
    void saveConfig();

public slots:
    void onComboThemeChanged(int index);
    
    void onCheckLegendVisible(int state);
    void onComboLegendAlignChanged(int index);
    void onSpinLegendFontSizeChanged(int i);
    void onSeriesEditClicked();
    void onComboTimeUnitChanged(int index);
    
    void onComboAxisChanged(int index);
    void onCheckAxisVisible(int state);
    void onCheckTitleVisible(int state);
    void onCheckLog(int state);
    void onSpinLogBaseChanged(int i);
    void onEditTitleChanged(const QString& text);
    void onEditTitleChanged2(const QString& text, int iAxis);
    void onSpinTitleSizeChanged(int i);
    void onSpinTitleSizeChanged2(int i, int iAxis);
    void onEditFormatChanged(const QString& text);
    void onSpinLabelSizeChanged(int i);
    void onSpinLabelSizeChanged2(int i, int iAxis);
    void onSpinMinChanged(double d);
    void onSpinMinChanged2(double d, int iAxis);
    void onSpinMaxChanged(double d);
    void onSpinMaxChanged2(double d, int iAxis);
    void onComboMinChanged(int index);
    void onComboMaxChanged(int index);
    void onSpinTicksChanged(int i);
    void onSpinMTicksChanged(int i);
    
    void onCheckAutoReload(int state);
    void onAutoReload(const QString &path);
    void onReloadClicked();
    void onSnapshotClicked();
    
    
private:
    struct AxisParam {
        AxisParam() : visible(true), title(true) {}
        
        bool visible, title;
        QString titleText;
        int titleSize, labelSize;
    };
    
    Ui::PlotterBoxChart *ui;
    QtCharts::QChartView *mChartView = nullptr;
    
    QVector<int> mBenchIdxs;
    const PlotParams mPlotParams;
    const QString mOrigFilename;
    const QVector<FileReload> mAddFilenames;
    const bool mAllIndexes;
    
    QFileSystemWatcher mWatcher;
    SeriesMapping mSeriesMapping;
    double mCurrentTimeFactor;      // from us
    AxisParam mAxesParams[2];
    bool mIgnoreEvents = false;
};


#endif // PLOTTER_BOXCHART_H
