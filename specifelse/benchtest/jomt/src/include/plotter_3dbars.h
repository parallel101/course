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

#ifndef PLOTTER_3DBARS_H
#define PLOTTER_3DBARS_H

#include "plot_parameters.h"
#include "series_dialog.h"

#include <QWidget>
#include <QVector>
#include <QString>
#include <QFileSystemWatcher>

namespace Ui {
class Plotter3DBars;
}
namespace QtDataVisualization {
class Q3DBars;
}
struct BenchResults;
struct FileReload;


class Plotter3DBars : public QWidget
{
    Q_OBJECT
    
public:
    explicit Plotter3DBars(const BenchResults &bchResults, const QVector<int> &bchIdxs,
                           const PlotParams &plotParams, const QString &filename,
                           const QVector<FileReload>& addFilenames, QWidget *parent = nullptr);
    ~Plotter3DBars();

private:
    void connectUI();
    void setupChart(const BenchResults &bchResults, const QVector<int> &bchIdxs, const PlotParams &plotParams, bool init = true);
    void setupOptions(bool init = true);
    void loadConfig(bool init);
    void saveConfig();

public slots:
    void onComboThemeChanged(int index);
    
    void onComboGradientChanged(int index);
    void onSpinThicknessChanged(double d);
    void onSpinFloorChanged(double d);
    void onSpinSpaceXChanged(double d);
    void onSpinSpaceZChanged(double d);
    void onSeriesEditClicked();
    void onComboTimeUnitChanged(int index);

    void onComboAxisChanged(int index);
    void onCheckAxisRotate(int state);
    void onCheckTitleVisible(int state);
    void onCheckLog(int state);
    void onSpinLogBaseChanged(int i);
    void onEditTitleChanged(const QString& text);
    void onEditTitleChanged2(const QString& text, int iAxis);
    void onEditFormatChanged(const QString& text);
    void onSpinMinChanged(double d);
    void onSpinMaxChanged(double d);
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
        AxisParam() : rotate(false), title(false), minIdx(0), maxIdx(0) {}
        void reset()
        {
            rotate = false;
            title = false;
            minIdx = 0;
            maxIdx = 0;
            titleText.clear();
            range.clear();
        }
        
        bool rotate, title;
        QString titleText;
        QStringList range;
        int minIdx, maxIdx;
    };
    void setupGradients();
    
    Ui::Plotter3DBars *ui;
    QtDataVisualization::Q3DBars *mBars;
    
    QVector<int> mBenchIdxs;
    const PlotParams mPlotParams;
    const QString mOrigFilename;
    const QVector<FileReload> mAddFilenames;
    const bool mAllIndexes;
    
    QFileSystemWatcher mWatcher;
    SeriesMapping mSeriesMapping;
    double mCurrentTimeFactor;      // from us
    AxisParam mAxesParams[3];
    QVector<QLinearGradient> mGrads;
    bool mIgnoreEvents = false;
};


#endif // PLOTTER_3DBARS_H
