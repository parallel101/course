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

#ifndef RESULT_SELECTOR_H
#define RESULT_SELECTOR_H

#include "benchmark_results.h"

#include <QSet>
#include <QVector>
#include <QString>
#include <QWidget>
#include <QFileSystemWatcher>

namespace Ui {
class ResultSelector;
}
class QTreeWidgetItem;


class ResultSelector : public QWidget
{
    Q_OBJECT
    
public:
    explicit ResultSelector(QWidget *parent = nullptr);
    explicit ResultSelector(const BenchResults &bchResults, const QString &fileName, QWidget *parent = nullptr);
    ~ResultSelector();
    
private:
    void connectUI();
    void loadConfig();
    void saveConfig();
    void updateComboBoxY();
    void updateResults(bool clear, const QSet<QString> unselected = {});
    
public slots:
    void onItemChanged(QTreeWidgetItem *item, int column);
    
    void onComboTypeChanged(int index);
    void onComboXChanged(int index);
    void onComboZChanged(int index);
    
    void onAutoReload(const QString &path);
    void updateReloadWatchList();
    void onCheckAutoReload(int state);
    void onReloadClicked();
    
    void onNewClicked();
    void onAppendClicked();
    void onOverwriteClicked();
    
    void onSelectAllClicked();
    void onSelectNoneClicked();
    
    void onPlotClicked();
    
private:
    Ui::ResultSelector *ui;
    
    BenchResults mBchResults;
    QString mOrigFilename;
    QVector<FileReload> mAddFilenames;
    
    QString mWorkingDir;
    QFileSystemWatcher mWatcher;
};


#endif // RESULT_SELECTOR_H
