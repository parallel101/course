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

#include "result_parser.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#define PARSE_DEBUG false
#if PARSE_DEBUG
  #include <QDebug>
#endif


// Find benchmark index by name
static int findExistingBenchmark(const BenchResults &bchResults, const QString &run_name)
{
    int idx = -1;
    for (int i=0; idx<0 && i<bchResults.benchmarks.size(); ++i)
        if (bchResults.benchmarks[i].run_name == run_name) idx = i;
    
    return idx;
}

// Remove aggregate suffix if any
static void cleanupName(BenchData &bchData)
{
    QString aggSuffix = "/repeats:";
    int lastIdx = bchData.run_name.lastIndexOf(aggSuffix);
    if (lastIdx > 0)
    {
        if (bchData.repetitions <= 0)
        {
            bool ok = false;
            int repetitions = bchData.run_name.midRef(lastIdx + aggSuffix.size()).toInt(&ok);
            if (ok)
                bchData.repetitions = repetitions;
        }
        bchData.run_name.truncate(lastIdx);
        bchData.name = bchData.run_name;
    }
}


// Parse benchmark results from json file
BenchResults ResultParser::parseJsonFile(const QString &filename, QString& errorMsg)
{
    BenchResults bchResults;
    
    // Read file
    QFile benchFile(filename);
    if ( !benchFile.open(QIODevice::ReadOnly) ) {
        errorMsg = "Couldn't open benchmark results file.";
        return bchResults;
    }
    QByteArray benchData = benchFile.readAll();
    benchFile.close();
    
    // Get Json main object
    QJsonDocument benchDoc( QJsonDocument::fromJson(benchData) );
    if (!benchDoc.isObject()) {
        errorMsg = "Not a json benchmark results file.";
        return bchResults;
    }
    QJsonObject benchObj = benchDoc.object();
    if (benchObj.isEmpty()) {
        errorMsg = "Empty json benchmark results file.";
        return bchResults;
    }
    
    
    /*
     * Context
     */
    if (benchObj.contains("context") && benchObj["context"].isObject())
    {
        QJsonObject ctxObj = benchObj["context"].toObject();
        
        // Meta
        if (ctxObj.contains("date") && ctxObj["date"].isString())
        {
            bchResults.context.date = ctxObj["date"].toString();
            if (PARSE_DEBUG) qDebug() << "date: " << bchResults.context.date;
        }
        if (ctxObj.contains("host_name") && ctxObj["host_name"].isString())
        {
            bchResults.context.host_name = ctxObj["host_name"].toString();
            if (PARSE_DEBUG) qDebug() << "host_name: " << bchResults.context.host_name;
        }
        if (ctxObj.contains("executable") && ctxObj["executable"].isString())
        {
            bchResults.context.executable = ctxObj["executable"].toString();
            if (PARSE_DEBUG) qDebug() << "executable: " << bchResults.context.executable;
        }
        
        // Build
        if (ctxObj.contains("library_build_type") && ctxObj["library_build_type"].isString())
        {
            bchResults.context.build_type = ctxObj["library_build_type"].toString();
            if (PARSE_DEBUG) qDebug() << "library_build_type: " << bchResults.context.build_type;
        }
        else if (ctxObj.contains("build_type") && ctxObj["build_type"].isString())
        {
            bchResults.context.build_type = ctxObj["build_type"].toString();
            if (PARSE_DEBUG) qDebug() << "build_type: " << bchResults.context.build_type;
        }
        
        // CPU
        if (ctxObj.contains("num_cpus") && ctxObj["num_cpus"].isDouble())
        {
            bchResults.context.num_cpus = ctxObj["num_cpus"].toInt();
            if (PARSE_DEBUG) qDebug() << "num_cpus: " << bchResults.context.num_cpus;
        }
        if (ctxObj.contains("mhz_per_cpu") && ctxObj["mhz_per_cpu"].isDouble())
        {
            bchResults.context.mhz_per_cpu = ctxObj["mhz_per_cpu"].toInt();
            if (PARSE_DEBUG) qDebug() << "mhz_per_cpu: " << bchResults.context.mhz_per_cpu;
        }
        if (ctxObj.contains("cpu_scaling_enabled") && ctxObj["cpu_scaling_enabled"].isBool())
        {
            bchResults.context.cpu_scaling_enabled = ctxObj["cpu_scaling_enabled"].toBool();
            if (PARSE_DEBUG) qDebug() << "cpu_scaling_enabled: " << bchResults.context.cpu_scaling_enabled;
        }
        
        // Caches
        if (ctxObj.contains("caches") && ctxObj["caches"].isArray())
        {
            QJsonArray cchArray = ctxObj["caches"].toArray();
            bchResults.context.caches.reserve( cchArray.size() );
            
            for (int cchIdx = 0; cchIdx < cchArray.size(); ++cchIdx)
            {
                // Cache
                QJsonObject cchObj = cchArray[cchIdx].toObject();
                BenchCache bchCache;
                if (PARSE_DEBUG) qDebug() << "Context cache";
                
                // Meta
                if (cchObj.contains("type") && cchObj["type"].isString())
                {
                    bchCache.type = cchObj["type"].toString();
                    if (PARSE_DEBUG) qDebug() << "-> type:" << bchCache.type;
                }
                if (cchObj.contains("level") && cchObj["level"].isDouble())
                {
                    bchCache.level = cchObj["level"].toInt();
                    if (PARSE_DEBUG) qDebug() << "-> level:" << bchCache.level;
                }
                if (cchObj.contains("size") && cchObj["size"].isDouble())
                {
                    bchCache.size = static_cast<int64_t>( cchObj["size"].toDouble() );
                    if (PARSE_DEBUG) qDebug() << "-> size:" << bchCache.size;
                }
                if (cchObj.contains("num_sharing") && cchObj["num_sharing"].isDouble())
                {
                    bchCache.num_sharing = cchObj["num_sharing"].toInt();
                   if (PARSE_DEBUG)  qDebug() << "-> num_sharing:" << bchCache.num_sharing;
                }
                
                //
                // Push bench cache
                bchResults.context.caches.append(bchCache);
                
                // New line between caches
                if (PARSE_DEBUG) qDebug() << "";
            }
        }
    }
    else
        qCritical() << "Results parsing: missing field 'context'";
    
    // New line between context and benchmarks
    if (PARSE_DEBUG) qDebug() << "";
    
    
    /*
     * Benchmarks
     */
    if (benchObj.contains("benchmarks") && benchObj["benchmarks"].isArray())
    {
        QJsonArray bchArray = benchObj["benchmarks"].toArray();
        bchResults.benchmarks.reserve( bchArray.size() );
        
        for (int bchIdx = 0; bchIdx < bchArray.size(); ++bchIdx)
        {
            // Benchmark
            QJsonObject bchObj = bchArray[bchIdx].toObject();
            BenchData bchData;
            
            //
            // Name
            if (bchObj.contains("name") && bchObj["name"].isString())
            {
                bchData.name = bchObj["name"].toString();
                if (PARSE_DEBUG) qDebug() << "bench name:" << bchData.name;
            }
            else {
                qCritical() << "Results parsing: missing benchmark field 'name'";
                continue;
            }
            // Run name
            if (bchObj.contains("run_name") && bchObj["run_name"].isString())
            {
                bchData.run_name = bchObj["run_name"].toString();
                if (PARSE_DEBUG) qDebug() << "-> run_name:" << bchData.run_name;
            }
            else {
                bchData.run_name = bchData.name;
                if (PARSE_DEBUG) qDebug() << "-> name as run_name:" << bchData.run_name;
            }
            cleanupName(bchData);
            // Run type
            if (bchObj.contains("run_type") && bchObj["run_type"].isString())
            {
                bchData.run_type = bchObj["run_type"].toString();
                if (PARSE_DEBUG) qDebug() << "-> run_type:" << bchData.run_type;
            }
            else {
                bchData.run_type = "iteration";
                if (PARSE_DEBUG) qDebug() << "-> default run_type:" << bchData.run_type;
            }
            
            //
            // Timing
            if (bchObj.contains("iterations") && bchObj["iterations"].isDouble())
            {
                bchData.iterations = bchObj["iterations"].toInt();
                if (PARSE_DEBUG) qDebug() << "-> iterations:" << bchData.iterations;
            }
            else {
                qCritical() << "Results parsing: missing benchmark field 'iterations'";
                continue;
            }
            
            if (bchObj.contains("real_time") && bchObj["real_time"].isDouble())
            {
                bchData.real_time.append( bchObj["real_time"].toDouble() );
                if (PARSE_DEBUG) qDebug() << "-> real_time:" << bchData.real_time.back();
            }
            else {
                qCritical() << "Results parsing: missing benchmark field 'real_time'";
                continue;
            }
            
            if (bchObj.contains("cpu_time") && bchObj["cpu_time"].isDouble())
            {
                bchData.cpu_time.append( bchObj["cpu_time"].toDouble() );
                if (PARSE_DEBUG) qDebug() << "-> cpu_time:" << bchData.cpu_time.back();
            }
            else {
                qCritical() << "Results parsing: missing benchmark field 'cpu_time'";
                continue;
            }
            
            if (bchObj.contains("time_unit") && bchObj["time_unit"].isString())
            {
                bchData.time_unit = bchObj["time_unit"].toString();
                if (PARSE_DEBUG) qDebug() << "-> time_unit:" << bchData.time_unit;
            }
            else {
                bchData.time_unit = "ns";
                if (PARSE_DEBUG) qDebug() << "-> default time_unit:" << bchData.time_unit;
            }
            // Time normalization (us)
            double timeFactor = 1.;
            if (bchData.time_unit == "ns")
            {
                timeFactor = 0.001;
                if (bchResults.meta.time_unit.isEmpty())     bchResults.meta.time_unit = "ns";
                else if (bchResults.meta.time_unit != "ns")  bchResults.meta.time_unit = "us";
            }
            else if (bchData.time_unit == "ms")
            {
                timeFactor = 1000.;
                if (bchResults.meta.time_unit.isEmpty())     bchResults.meta.time_unit = "ms";
                else if (bchResults.meta.time_unit != "ms")  bchResults.meta.time_unit = "us";
                
            }
            else {
                bchResults.meta.time_unit = "us";
            }
            bchData.real_time_us = bchData.real_time.back() * timeFactor;
            bchData.cpu_time_us  = bchData.cpu_time.back()  * timeFactor;
            
            //
            // Throughput
            if (bchObj.contains("bytes_per_second") && bchObj["bytes_per_second"].isDouble())
            {
                bchData.kbytes_sec.append(bchObj["bytes_per_second"].toDouble() * 0.001);
                bchData.kbytes_sec_dflt = bchData.kbytes_sec.back();
                bchResults.meta.hasBytesSec = true;
                if (PARSE_DEBUG) qDebug() << "-> kbytes_sec:" << bchData.kbytes_sec_dflt;
            }
            if (bchObj.contains("items_per_second") && bchObj["items_per_second"].isDouble())
            {
                bchData.kitems_sec.append(bchObj["items_per_second"].toDouble() * 0.001);
                bchData.kitems_sec_dflt = bchData.kitems_sec.back();
                bchResults.meta.hasItemsSec = true;
                if (PARSE_DEBUG) qDebug() << "-> kitems_sec:" << bchData.kitems_sec_dflt;
            }
            
            
            /*
             * Existing benchmark
             */
            int idx = findExistingBenchmark(bchResults, bchData.run_name);
            if (idx >= 0)
            {
                BenchData &exBchData = bchResults.benchmarks[idx];
                
                /*
                 * Aggregate type
                 */
                if (bchData.run_type == "aggregate")
                {
                    if (PARSE_DEBUG) qDebug() << "-> append aggregate:" << exBchData.name;
                    
                    // Name
                    QString aggregate_name;
                    if (bchObj.contains("aggregate_name") && bchObj["aggregate_name"].isString())
                    {
                        aggregate_name = bchObj["aggregate_name"].toString();
                        if (PARSE_DEBUG) qDebug() << "-> aggregate_name:" << aggregate_name;
                    }
                    else {
                        qCritical() << "Results parsing: missing benchmark field 'aggregate_name'";
                        continue;
                    }
                    // Type
                    if (aggregate_name == "mean") {
                        exBchData.mean_cpu  = bchData.cpu_time_us;
                        exBchData.mean_real = bchData.real_time_us;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            exBchData.mean_kbytes = bchData.kbytes_sec_dflt;
                        if ( !bchData.kitems_sec.isEmpty() )
                            exBchData.mean_kitems = bchData.kitems_sec_dflt;
                    }
                    else if (aggregate_name == "median") {
                        exBchData.median_cpu  = bchData.cpu_time_us;
                        exBchData.median_real = bchData.real_time_us;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            exBchData.median_kbytes = bchData.kbytes_sec_dflt;
                        if ( !bchData.kitems_sec.isEmpty() )
                            exBchData.median_kitems = bchData.kitems_sec_dflt;
                    }
                    else if (aggregate_name == "stddev") {
                        exBchData.stddev_cpu  = bchData.cpu_time_us;
                        exBchData.stddev_real = bchData.real_time_us;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            exBchData.stddev_kbytes = bchData.kbytes_sec_dflt;
                        if ( !bchData.kitems_sec.isEmpty() )
                            exBchData.stddev_kitems = bchData.kitems_sec_dflt;
                    }
                    else if (aggregate_name == "cv") {
                        exBchData.cv_cpu  = bchData.cpu_time.back()  * 100;  // percent
                        exBchData.cv_real = bchData.real_time.back() * 100;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            exBchData.cv_kbytes = bchData.kbytes_sec_dflt * 100;
                        if ( !bchData.kitems_sec.isEmpty() )
                            exBchData.cv_kitems = bchData.kitems_sec_dflt * 100;
                        bchResults.meta.hasCv = true;
                    }
                    else {
                        qCritical() << "Results parsing: unknown benchmark value for 'aggregate_name' ->" << aggregate_name;
                        continue;
                    }
                    
                    // New aggregate line
                    if (PARSE_DEBUG) qDebug() << "||";
                }
                
                /*
                 * Iteration type (from aggregate)
                 */
                else
                {
                    if (PARSE_DEBUG) qDebug() << "-> append iteration:" << exBchData.name;
                    
                    // Append data
                    exBchData.cpu_time.append( bchData.cpu_time.back() );
                    exBchData.cpu_time_us = std::min(exBchData.cpu_time_us, bchData.cpu_time_us);
                    
                    exBchData.real_time.append( bchData.real_time.back() );
                    exBchData.real_time_us = std::min(exBchData.real_time_us, bchData.real_time_us);
                    
                    if ( !bchData.kbytes_sec.isEmpty() ) {
                        exBchData.kbytes_sec.append( bchData.kbytes_sec_dflt );
                        exBchData.kbytes_sec_dflt = std::min(exBchData.kbytes_sec_dflt, bchData.kbytes_sec_dflt);
                    }
                    if ( !bchData.kitems_sec.isEmpty() ) {
                        exBchData.kitems_sec.append( bchData.kitems_sec_dflt );
                        exBchData.kitems_sec_dflt = std::min(exBchData.kitems_sec_dflt, bchData.kitems_sec_dflt);
                    }
                    
                    // Min/Max
                    if (!exBchData.hasAggregate) //First -> init
                    {
                        exBchData.min_cpu  = exBchData.cpu_time_us;
                        exBchData.max_cpu  = std::max(exBchData.cpu_time_us,  bchData.cpu_time_us);
    
                        exBchData.min_real = exBchData.real_time_us;
                        exBchData.max_real = std::max(exBchData.real_time_us, bchData.real_time_us);
                        
                        if ( !bchData.kbytes_sec.isEmpty() ) {
                            exBchData.min_kbytes = exBchData.kbytes_sec_dflt;
                            exBchData.max_kbytes = std::max(exBchData.kbytes_sec_dflt, bchData.kbytes_sec_dflt);
                        }
                        if ( !bchData.kitems_sec.isEmpty() ) {
                            exBchData.min_kitems = exBchData.kitems_sec_dflt;
                            exBchData.max_kitems = std::max(exBchData.kitems_sec_dflt, bchData.kitems_sec_dflt);
                        }
                    }
                    else
                    {
                        if (exBchData.min_cpu  > bchData.cpu_time_us)  exBchData.min_cpu = bchData.cpu_time_us;
                        if (exBchData.max_cpu  < bchData.cpu_time_us)  exBchData.max_cpu = bchData.cpu_time_us;
                        
                        if (exBchData.min_real > bchData.real_time_us) exBchData.min_real = bchData.real_time_us;
                        if (exBchData.max_real < bchData.real_time_us) exBchData.max_real = bchData.real_time_us;
                        
                        if ( !bchData.kbytes_sec.isEmpty() ) {
                            if (exBchData.min_kbytes > bchData.kbytes_sec_dflt) exBchData.min_kbytes = bchData.kbytes_sec_dflt;
                            if (exBchData.max_kbytes < bchData.kbytes_sec_dflt) exBchData.max_kbytes = bchData.kbytes_sec_dflt;
                        }
                        if ( !bchData.kitems_sec.isEmpty() ) {
                            if (exBchData.min_kitems > bchData.kitems_sec_dflt) exBchData.min_kitems = bchData.kitems_sec_dflt;
                            if (exBchData.max_kitems < bchData.kitems_sec_dflt) exBchData.max_kitems = bchData.kitems_sec_dflt;
                        }
                    }
                    
                    // State
                    exBchData.hasAggregate = true;
                    bchResults.meta.hasAggregate = true;
                    bchResults.meta.onlyAggregate = false;
                    
                    // Debug
                    if (PARSE_DEBUG) {
                        qDebug() << "** exBchData.min_cpu:"  << exBchData.min_cpu;
                        qDebug() << "** exBchData.max_cpu:"  << exBchData.max_cpu;
                        qDebug() << "** exBchData.min_real:" << exBchData.min_real;
                        qDebug() << "** exBchData.max_real:" << exBchData.max_real;
                        if ( !exBchData.kbytes_sec.isEmpty() ) {
                            qDebug() << "** exBchData.min_kbytes:" << exBchData.min_kbytes;
                            qDebug() << "** exBchData.max_kbytes:" << exBchData.max_kbytes;
                        }
                        if ( !exBchData.kitems_sec.isEmpty() ) {
                            qDebug() << "** exBchData.min_kitems:" << exBchData.min_kitems;
                            qDebug() << "** exBchData.max_kitems:" << exBchData.max_kitems;
                        }
                    }
                    
                    // New  append line
                    if (PARSE_DEBUG) qDebug() << "|";
                }
            }
            
            /*
             * New benchmark
             */
            else
            {
                /*
                 * Aggregate-only type
                 */
                if (bchData.run_type == "aggregate")
                {
                    if (PARSE_DEBUG) qDebug() << "-> new aggregate-only";
                    
                    // Name
                    QString aggregate_name;
                    if (bchObj.contains("aggregate_name") && bchObj["aggregate_name"].isString())
                    {
                        aggregate_name = bchObj["aggregate_name"].toString();
                        if (PARSE_DEBUG) qDebug() << "-> aggregate_name:" << aggregate_name;
                    }
                    else {
                        qCritical() << "Results parsing: missing benchmark field 'aggregate_name'";
                        continue;
                    }
                    // Type
                    if (aggregate_name == "mean") {
                        bchData.mean_cpu  = bchData.cpu_time_us;
                        bchData.mean_real = bchData.real_time_us;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            bchData.mean_kbytes = bchData.kbytes_sec_dflt;
                        if ( !bchData.kitems_sec.isEmpty() )
                            bchData.mean_kitems = bchData.kitems_sec_dflt;
                    }
                    else if (aggregate_name == "median") {
                        bchData.median_cpu  = bchData.cpu_time_us;
                        bchData.median_real = bchData.real_time_us;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            bchData.median_kbytes = bchData.kbytes_sec_dflt;
                        if ( !bchData.kitems_sec.isEmpty() )
                            bchData.median_kitems = bchData.kitems_sec_dflt;
                    }
                    else if (aggregate_name == "stddev") {
                        bchData.stddev_cpu  = bchData.cpu_time_us;
                        bchData.stddev_real = bchData.real_time_us;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            bchData.stddev_kbytes = bchData.kbytes_sec_dflt;
                        if ( !bchData.kitems_sec.isEmpty() )
                            bchData.stddev_kitems = bchData.kitems_sec_dflt;
                    }
                    else if (aggregate_name == "cv") {
                        bchData.cv_cpu  = bchData.cpu_time.back()  * 100;  // percent
                        bchData.cv_real = bchData.real_time.back() * 100;
                        if ( !bchData.kbytes_sec.isEmpty() )
                            bchData.cv_kbytes = bchData.kbytes_sec_dflt * 100;
                        if ( !bchData.kitems_sec.isEmpty() )
                            bchData.cv_kitems = bchData.kitems_sec_dflt * 100;
                        bchResults.meta.hasCv = true;
                    }
                    else {
                        qCritical() << "Results parsing: unknown benchmark value for 'aggregate_name' ->" << aggregate_name;
                        continue;
                    }
                    
                    // Init
                    bchData.hasAggregate = true;
                    bchResults.meta.hasAggregate = true;
                    
                    bchData.cpu_time_us  = -1;
                    bchData.real_time_us = -1;
                    bchData.min_cpu  = bchData.max_cpu  = -1;
                    bchData.min_real = bchData.max_real = -1;
                }
                
                /*
                 * Add new benchmark
                 */
                // Arguments (extract from 'run_name')
                bchData.arguments = bchData.run_name.split('/');
                QString bchName = bchData.arguments.front();
                bchData.arguments.pop_front();
                
                // Debug: params
                for (int prmIdx = 0; prmIdx < bchData.arguments.size(); ++prmIdx)
                    if (PARSE_DEBUG) qDebug() << "-> param[" << prmIdx << "]:" << bchData.arguments[prmIdx];
                
                // Templates (extract from 'run_name' too)
                int tpltIdx = bchName.indexOf("<");
                if (tpltIdx > 0)
                {
                    int tpltLast = bchName.lastIndexOf(">");
                    if (tpltLast != bchName.size()-1) {
                        qCritical() << "Bad benchmark template formatting:" << bchName;
                        continue;
                    }
                    QString tpltName = bchName.mid(tpltIdx+1, tpltLast-tpltIdx-1);
                    
                    // Split
                    int startIdx = 0;
                    int commaIdx = tpltName.indexOf(",");
                    while (commaIdx > 0)
                    {
                        QString leftString = tpltName.left(commaIdx);
                        int open = leftString.count('<');
                        int close = leftString.count('>');
                        
                        if (open <= close)
                        {
                            bchData.templates.append( tpltName.left(commaIdx).trimmed() );
                            tpltName.remove(0, commaIdx+1);
                            startIdx = 0;
                        }
                        else {
                            startIdx = commaIdx+1;
                        }
                        commaIdx = tpltName.indexOf(",", startIdx);
                    }
                    // Last
                    bchData.templates.append( tpltName.trimmed() );
                    
                    // For base name
                    bchName.truncate(tpltIdx);
                }
                // Debug: templates
                for (int idx = 0; idx < bchData.templates.size(); ++idx)
                    if (PARSE_DEBUG) qDebug() << "-> template[" << idx << "]:" << bchData.templates[idx];
                
                // Base name (i.e. name without templates/arguments)
                bchData.base_name = bchName;
                if (PARSE_DEBUG) qDebug() << "-> base_name:" << bchData.base_name;
                
                // JOMT
                // Family / Container
                if ( bchData.base_name.startsWith("JOMT_") )
                {
                    // Examples: "JOMT_Fill_vector<int>/64" Vs "JOMT_Fill_deque<int>/64"
                    bchData.base_name = bchData.base_name.remove(0,5);  //remove prefix
                    int idx = bchData.base_name.indexOf('_');
                    if (idx > 0)
                    {
                        bchData.family = bchData.base_name.left(idx);
                        bchData.container = bchData.base_name;
                        bchData.container = bchData.container.remove(0,idx+1);
                    }
                }
                // Classic (base name as family name)
                else
                    bchData.family = bchData.base_name;
                
                if (PARSE_DEBUG) qDebug() << "-> family:" << bchData.family;
                if (PARSE_DEBUG) qDebug() << "-> container:" << bchData.container;
    
    
                //
                // Meta
                if (bchObj.contains("repetitions") && bchObj["repetitions"].isDouble())
                {
                    bchData.repetitions = bchObj["repetitions"].toInt();
                    if (PARSE_DEBUG) qDebug() << "-> repetitions:" << bchData.repetitions;
                }
                if (bchObj.contains("repetition_index") && bchObj["repetition_index"].isDouble())
                {
                    bchData.repetition_index = bchObj["repetition_index"].toInt();
                    if (PARSE_DEBUG) qDebug() << "-> repetition_index:" << bchData.repetition_index;
                }
                if (bchObj.contains("threads") && bchObj["threads"].isDouble())
                {
                    bchData.threads = bchObj["threads"].toInt();
                    if (PARSE_DEBUG) qDebug() << "-> threads:" << bchData.threads;
                }
                
                //
                // Global Meta
                if (bchData.arguments.size() > bchResults.meta.maxArguments)
                    bchResults.meta.maxArguments = bchData.arguments.size();
                if (bchData.templates.size() > bchResults.meta.maxTemplates)
                    bchResults.meta.maxTemplates = bchData.templates.size();
                bchResults.meta.onlyAggregate &= bchData.min_real < 0.;
                
                //
                // Push new BenchData
                bchResults.benchmarks.append(bchData);
                
                // New line between benchmarks
                if (PARSE_DEBUG) qDebug() << "";
            }
        }
    }
    else
        qCritical() << "Results parsing: missing field 'benchmarks'";
    
    // Debug
    if (PARSE_DEBUG) {
        qDebug() << "meta.maxArgs:"         << bchResults.meta.maxArguments;
        qDebug() << "meta.maxTemplates:"    << bchResults.meta.maxTemplates;
        qDebug() << "meta.hasAggregate:"    << bchResults.meta.hasAggregate;
    }
    
    
    return bchResults;
}
