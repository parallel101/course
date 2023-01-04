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

#ifndef SERIES_DIALOG_H
#define SERIES_DIALOG_H

#include <QDialog>
#include <QVector>
#include <QString>
#include <QColor>

namespace Ui {
class SeriesDialog;
}

struct SeriesConfig {
    SeriesConfig(const QString &oldName_, const QString &newName_)
        : oldName(oldName_)
        , newName(newName_)
    {}
    
    QString oldName, newName;
    QColor oldColor, newColor;
};
inline bool operator==(const SeriesConfig& lhs, const SeriesConfig& rhs) {
    return (lhs.oldName == rhs.oldName);
}
typedef QVector<SeriesConfig>   SeriesMapping;


class SeriesDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit SeriesDialog(const SeriesMapping &mapping, QWidget *parent = nullptr);
    ~SeriesDialog();
    
    const SeriesMapping& getMapping() { return mMapping; }
    
public slots:
    virtual void accept();
    void onRestoreClicked();
    
private:
    Ui::SeriesDialog *ui;
    
    SeriesMapping mMapping;
};


#endif // SERIES_DIALOG_H
