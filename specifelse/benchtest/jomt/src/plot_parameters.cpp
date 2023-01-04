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

#include "plot_parameters.h"

#include <algorithm>

const char* config_folder = "jomtSettings/";


double getYPlotValue(const BenchData &bchData, PlotValueType yType)
{
    switch (yType)
    {
        // CPU time
        case CpuTimeType: {
            return bchData.cpu_time_us;
        }
        case CpuTimeMinType: {
            return bchData.min_cpu;
        }
        case CpuTimeMeanType: {
            return bchData.mean_cpu;
        }
        case CpuTimeMedianType: {
            return bchData.median_cpu;
        }
        case CpuTimeStddevType: {
            return bchData.stddev_cpu;
        }
        case CpuTimeCvType: {
            return bchData.cv_cpu;
        }
        
        // Real time
        case RealTimeType: {
            return bchData.real_time_us;
        }
        case RealTimeMinType: {
            return bchData.min_real;
        }
        case RealTimeMeanType: {
            return bchData.mean_real;
        }
        case RealTimeMedianType: {
            return bchData.median_real;
        }
        case RealTimeStddevType: {
            return bchData.stddev_real;
        }
        case RealTimeCvType: {
            return bchData.cv_real;
        }
        
        // Iterations
        case IterationsType: {
            return bchData.iterations;
        }
        
        // Bytes/s
        case BytesType: {
            return bchData.kbytes_sec_dflt;
        }
        case BytesMinType: {
            return bchData.min_kbytes;
        }
        case BytesMeanType: {
            return bchData.mean_kbytes;
        }
        case BytesMedianType: {
            return bchData.median_kbytes;
        }
        case BytesStddevType: {
            return bchData.stddev_kbytes;
        }
        case BytesCvType: {
            return bchData.cv_kbytes;
        }
        
        // Items/s
        case ItemsType: {
            return bchData.kitems_sec_dflt;
        }
        case ItemsMinType: {
            return bchData.min_kitems;
        }
        case ItemsMeanType: {
            return bchData.mean_kitems;
        }
        case ItemsMedianType: {
            return bchData.median_kitems;
        }
        case ItemsStddevType: {
            return bchData.stddev_kitems;
        }
        case ItemsCvType: {
            return bchData.cv_kitems;
        }
    }
    
    return -1;
}

QString getYPlotName(PlotValueType yType, QString timeUnit)
{
    if (!timeUnit.isEmpty())
        timeUnit = " (" + timeUnit + ")";
    
    switch (yType)
    {
        // CPU time
        case CpuTimeType: {
            return "CPU time" + timeUnit;
        }
        case CpuTimeMinType: {
            return "CPU min time" + timeUnit;
        }
        case CpuTimeMeanType: {
            return "CPU mean time" + timeUnit;
        }
        case CpuTimeMedianType: {
            return "CPU median time" + timeUnit;
        }
        case CpuTimeStddevType: {
            return "CPU stddev time" + timeUnit;
        }
        case CpuTimeCvType: {
            return "CPU cv (%)";
        }
        
        // Real time
        case RealTimeType: {
            return "Real time" + timeUnit;
        }
        case RealTimeMinType: {
            return "Real min time" + timeUnit;
        }
        case RealTimeMeanType: {
            return "Real mean time" + timeUnit;
        }
        case RealTimeMedianType: {
            return "Real median time" + timeUnit;
        }
        case RealTimeStddevType: {
            return "Real stddev time" + timeUnit;
        }
        case RealTimeCvType: {
            return "Real cv (%)";
        }
        
        // Iterations
        case IterationsType: {
            return "Iterations";
        }
        
        // Bytes/s
        case BytesType: {
            return "Bytes/s (k)";
        }
        case BytesMinType: {
            return "Bytes/s min (k)";
        }
        case BytesMeanType: {
            return "Bytes/s mean (k)";
        }
        case BytesMedianType: {
            return "Bytes/s median (k)";
        }
        case BytesStddevType: {
            return "Bytes/s stddev (k)";
        }
        case BytesCvType: {
            return "Bytes/s cv (%)";
        }
        
        // Items/s
        case ItemsType: {
            return "Items/s (k)";
        }
        case ItemsMinType: {
            return "Items/s min (k)";
        }
        case ItemsMeanType: {
            return "Items/s mean (k)";
        }
        case ItemsMedianType: {
            return "Items/s median (k)";
        }
        case ItemsStddevType: {
            return "Items/s stddev (k)";
        }
        case ItemsCvType: {
            return "Items/s cv (%)";
        }
    }
    
    return "Unknown";
}

double normalizeTimeUs(const BenchData &bchData, double value)
{
    double timeFactor = 1.;
    if      (bchData.time_unit == "ns") timeFactor = 0.001;
    else if (bchData.time_unit == "ms") timeFactor = 1000.;
    return value * timeFactor;
}

bool isYTimeBased(PlotValueType yType)
{
    if (   yType != PlotValueType::RealTimeType       && yType != PlotValueType::CpuTimeType
        && yType != PlotValueType::RealTimeMinType    && yType != PlotValueType::CpuTimeMinType
        && yType != PlotValueType::RealTimeMeanType   && yType != PlotValueType::CpuTimeMeanType
        && yType != PlotValueType::RealTimeMedianType && yType != PlotValueType::CpuTimeMedianType
        && yType != PlotValueType::RealTimeStddevType && yType != PlotValueType::CpuTimeStddevType )
        return false;
    
    return true;
}

double findMedian(QVector<double> sorted, int begin, int end)
{
    int count = end - begin;
    if (count <= 0) return 0.;
    
    if (count % 2) {
        return sorted.at(count / 2 + begin);
    } else {
        qreal right = sorted.at(count / 2 + begin);
        qreal left = sorted.at(count / 2 - 1 + begin);
        return (right + left) / 2.0;
    }
}

BenchYStats getYPlotStats(BenchData &bchData, PlotValueType yType)
{
    BenchYStats statRes;
    
    // No statistics
    if (!bchData.hasAggregate) {
        statRes.min      = 0.;
        statRes.max      = 0.;
        statRes.median   = 0.;
        statRes.lowQuart = 0.;
        statRes.uppQuart = 0.;
        
        return statRes;
    }
    
    switch (yType)
    {
        // CPU time
        case CpuTimeType:
        case CpuTimeMinType: case CpuTimeMeanType: case CpuTimeMedianType: case CpuTimeStddevType:
        {
            statRes.min    = bchData.min_cpu;
            statRes.max    = bchData.max_cpu;
            statRes.median = bchData.median_cpu;
            
            std::sort(bchData.cpu_time.begin(), bchData.cpu_time.end());
            int count = bchData.cpu_time.count();
            statRes.lowQuart = normalizeTimeUs(bchData, findMedian(bchData.cpu_time, 0, count/2));
            statRes.uppQuart = normalizeTimeUs(bchData, findMedian(bchData.cpu_time, count/2 + (count%2), count));
            
            break;
        }
        // Real time
        case RealTimeType:
        case RealTimeMinType: case RealTimeMeanType: case RealTimeMedianType: case RealTimeStddevType:
        {
            statRes.min    = bchData.min_real;
            statRes.max    = bchData.max_real;
            statRes.median = bchData.median_real;
            
            std::sort(bchData.real_time.begin(), bchData.real_time.end());
            int count = bchData.real_time.count();
            statRes.lowQuart = normalizeTimeUs(bchData, findMedian(bchData.real_time, 0, count/2));
            statRes.uppQuart = normalizeTimeUs(bchData, findMedian(bchData.real_time, count/2 + (count%2), count));
            
            break;
        }
        // Bytes/s
        case BytesType:
        case BytesMinType: case BytesMeanType: case BytesMedianType: case BytesStddevType:
        {
            statRes.min    = bchData.min_kbytes;
            statRes.max    = bchData.max_kbytes;
            statRes.median = bchData.median_kbytes;
            
            std::sort(bchData.kbytes_sec.begin(), bchData.kbytes_sec.end());
            int count = bchData.kbytes_sec.count();
            statRes.lowQuart = findMedian(bchData.kbytes_sec, 0, count/2);
            statRes.uppQuart = findMedian(bchData.kbytes_sec, count/2 + (count%2), count);
            
            break;
        }
        // Items/s
        case ItemsType:
        case ItemsMinType: case ItemsMeanType: case ItemsMedianType: case ItemsStddevType:
        {
            statRes.min    = bchData.min_kitems;
            statRes.max    = bchData.max_kitems;
            statRes.median = bchData.median_kitems;
            
            std::sort(bchData.kitems_sec.begin(), bchData.kitems_sec.end());
            int count = bchData.kitems_sec.count();
            statRes.lowQuart = findMedian(bchData.kitems_sec, 0, count/2);
            statRes.uppQuart = findMedian(bchData.kitems_sec, count/2 + (count%2), count);
            
            break;
        }
        default:    //Error
        {
            statRes.min      = 0.;
            statRes.max      = 0.;
            statRes.median   = 0.;
            statRes.lowQuart = 0.;
            statRes.uppQuart = 0.;
            
            break;
        }
    }
    
    return statRes;
}

bool commonPartEqual(const QStringList &listA, const QStringList &listB)
{
    bool isEqual = true;
    int maxIdx = std::min(listA.size(), listB.size());
    if (maxIdx <= 0) return false;
    
    for (int idx=0; isEqual && idx<maxIdx; ++idx)
        isEqual = listA[idx] == listB[idx];
    
    return isEqual;
}

bool sameResultsFiles(const QString &fileA, const QString &fileB,
                      const QVector<FileReload> &addFilesA, const QVector<FileReload> &addFilesB)
{
    if (fileA != fileB)
        return false;
    if (addFilesA.size() != addFilesB.size())
        return false;
    
    for (int i=0; i<addFilesA.size(); ++i) {
        if ( !addFilesB.contains(addFilesA[i]) )
            return false;
    }
    
    return true;
}
