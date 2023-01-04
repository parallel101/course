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

#include "benchmark_results.h"
#include "result_parser.h"
#include "plot_parameters.h"
#include "commandline_handler.h"
#include "result_selector.h"

#include <QDir>
#include <QIcon>
#include <QApplication>
#include <QScopedPointer>
#include <QDebug>

#define APP_NAME    "JOMT"
#define APP_VER     "1.0b"
#define APP_ICON    ":/jomt_icon.png"

// Debug
#define DEFAULT_DIR   ""
#define DEFAULT_FILE  ""


int main(int argc, char *argv[])
{
    // Init
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(APP_NAME);
    QCoreApplication::setApplicationVersion(APP_VER);
    QApplication::setWindowIcon( QIcon(APP_ICON) );
    
    QDir configDir(config_folder);
    if (!configDir.exists())
        configDir.mkpath(".");
    
    //
    // Command line options
    CommandLineHandler cmdHandler;
    bool isCmd = cmdHandler.process(app);
    
    QScopedPointer<ResultSelector> resultSelector;
    if (!isCmd)
    {
        // Debug test
        QString fileName(DEFAULT_FILE);
        if (!QString(DEFAULT_DIR).isEmpty() && !fileName.isEmpty())
        {
            QDir jmtDir(DEFAULT_DIR);
            
            QString errorMsg;
            BenchResults bchResults = ResultParser::parseJsonFile( jmtDir.filePath(fileName), errorMsg );
            
            if ( bchResults.benchmarks.isEmpty() ) {
                qCritical() << "Error parsing file: " << fileName << " -> " << errorMsg;
                return 1;
            }
            // Selector Test
            resultSelector.reset(new ResultSelector(bchResults, jmtDir.filePath(fileName)));
        }
        else
            // Show empty selector
            resultSelector.reset(new ResultSelector());
        resultSelector->show();
    }
    
    //
    // Execute
    return app.exec();
}
