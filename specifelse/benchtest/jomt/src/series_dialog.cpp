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

#include "series_dialog.h"
#include "ui_series_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QHBoxLayout>
#include <QColorDialog>
#include <QScreen>
#include <QGuiApplication>


class FieldWidget : public QWidget
{
public:
    QLineEdit nameEdit;
    QPushButton colorButton;
    QColor colorValue;
    
    FieldWidget(const QString& name, const QColor& color, QWidget *parent)
        : QWidget(parent)
        , nameEdit(name, this)
        , colorButton(this)
        , colorValue(color)
    {
        // Name
        nameEdit.setCursorPosition(0);
        if (nameEdit.text().isEmpty())
            nameEdit.setEnabled(false);
        
        // Color
        colorButton.setStyleSheet( "QPushButton { background-color: " + colorValue.name() + "; }"
                                 + "QPushButton:hover:!pressed { border-style: inset; border-width: 3px; }" );
        colorButton.setMinimumHeight( std::max(20, nameEdit.height()) );
        colorButton.setMinimumWidth(colorButton.minimumHeight() * 1.5);
        colorButton.setFixedSize(colorButton.minimumWidth(), colorButton.minimumHeight());
        colorButton.setToolTip("Change color");
        
        // Connect
        connect(&colorButton, &QPushButton::clicked, this, &FieldWidget::onColorClicked);
        
        // Layout
        QHBoxLayout *layout = new QHBoxLayout;
        layout->addWidget(&nameEdit);
        layout->addWidget(&colorButton);
        layout->setContentsMargins(0,0,0,0);
        setLayout(layout);
    }
    
public slots:
    void onColorClicked()
    {
        QColor newColor = QColorDialog::getColor(colorValue, this, nameEdit.text());
        if (newColor.isValid() && newColor != colorValue)
        {
            colorValue = newColor;
            colorButton.setStyleSheet("QPushButton { background-color: " + colorValue.name() + "; }");
        }
    }
};


SeriesDialog::SeriesDialog(const SeriesMapping &mapping, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::SeriesDialog)
    , mMapping(mapping)
{
    ui->setupUi(this);
    this->setWindowTitle( "Edit series" );
    
    // Setup form
    ui->formLayout->addRow("<b>Original:</b>", new QLabel("<b>Modified:</b>", this));
    
    for (const auto& config : qAsConst(mMapping)) {
        ui->formLayout->addRow(config.oldName.isEmpty() ? "<no-name>" : config.oldName,
                               new FieldWidget(config.newName, config.newColor, this));
    }
    
    // Connect
    connect(ui->buttonBox->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked, this, &SeriesDialog::onRestoreClicked);
    
    // Default size
    QSize size = this->size();
    QSize newSize = QGuiApplication::primaryScreen()->size();
    newSize *= 0.25f;
    if (newSize.width() > size.width())
        resize(newSize.width(), size.height());
}

SeriesDialog::~SeriesDialog()
{
    delete ui;
}


void SeriesDialog::accept()
{
    // Save edited
    for (int idx = 0; idx < mMapping.size(); ++idx)
    {
        auto item = ui->formLayout->itemAt(idx + 1, QFormLayout::FieldRole);
        auto fieldWidget = dynamic_cast<FieldWidget*>(item->widget());
        
        if (!fieldWidget->nameEdit.text().isEmpty()) {
            mMapping[idx].newName = fieldWidget->nameEdit.text();
        }
        mMapping[idx].newColor = fieldWidget->colorValue;
    }
    
    QDialog::accept();
}

void SeriesDialog::onRestoreClicked()
{
    for (int idx = 0; idx < mMapping.size(); ++idx)
    {
        auto item = ui->formLayout->itemAt(idx + 1, QFormLayout::FieldRole);
        auto fieldWidget = dynamic_cast<FieldWidget*>(item->widget());
        
        fieldWidget->nameEdit.setText( mMapping[idx].oldName );
        fieldWidget->nameEdit.setCursorPosition(0);
        fieldWidget->colorValue = mMapping[idx].oldColor;
        fieldWidget->colorButton.setStyleSheet( "QPushButton { background-color: " + fieldWidget->colorValue.name() + "; }"
                                              + "QPushButton:hover:!pressed { border-style: inset; border-width: 3px; }" );
    }
}
