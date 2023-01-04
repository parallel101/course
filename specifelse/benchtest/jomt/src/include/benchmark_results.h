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

#ifndef BENCHMARK_DATA_H
#define BENCHMARK_DATA_H

#include <QString>
#include <QStringList>
#include <QVector>


/*
 * Structures
 */
// Additional file
struct FileReload {
    QString filename;
    bool isAppend;
};
inline bool operator==(const FileReload& lhs, const FileReload& rhs) {
    return (lhs.isAppend == rhs.isAppend && lhs.filename == rhs.filename);
}

// Context Cache
struct BenchCache {
    QString type;
    int level;
    int64_t size;
    int num_sharing;
};

// Benchmarks Context
struct BenchContext {
    QString date;
    QString host_name;
    QString executable;
    int num_cpus;
    int mhz_per_cpu;
    bool cpu_scaling_enabled;
    //"load_avg": [], // ?
    QString build_type;
    QVector<BenchCache> caches;
};

// Benchmark Data
struct BenchData {
    // Iterations
    QString name;
    QString run_name;
    QString run_type;
    int repetitions = 0;
    int repetition_index;
    int threads;
    int iterations;
    QString time_unit;
    QVector<double> real_time; // One per iteration
    QVector<double> cpu_time;
    QVector<double> kbytes_sec;
    QVector<double> kitems_sec;
    
    // Aggregate (all durations in us/cv in %)
    bool hasAggregate = false;
    double min_real, min_cpu, min_kbytes, min_kitems;
    double max_real, max_cpu, max_kbytes, max_kitems;
    double mean_real, mean_cpu, mean_kbytes, mean_kitems;
    double median_real, median_cpu, median_kbytes, median_kitems;
    double stddev_real, stddev_cpu, stddev_kbytes, stddev_kitems;
    double cv_real = -1, cv_cpu = -1, cv_kbytes = -1, cv_kitems = -1;
    
    // Meta
    // Note: JOMT format = "JOMT_FamilyName_ContainerName<templates>/params
    QString base_name;      // run_name without template/param/JOMT prefix
    QString family;         // family/algo name (empty if not JOMT)
    QString container;      // container name (empty if not JOMT)
    QStringList arguments;  // benchmark arguments
    QStringList templates;  // template parameters
    
    // Default (use associated 'min' values if has aggregate)
    double real_time_us, cpu_time_us;   // in us
    double kbytes_sec_dflt = 0, kitems_sec_dflt = 0;
};

// Benchmark Subset
struct BenchSubset {
    BenchSubset() {}
    BenchSubset(const QString &name_) : name(name_) {}
    BenchSubset(const QString &name_, int idx) : name(name_)
    { idxs.push_back(idx); }
    
    // Data
    QString name;
    QVector<int> idxs;
};

// Benchmark Meta
struct BenchMeta {
    // Data
    bool hasAggregate = false, onlyAggregate = true, hasCv = false;
    bool hasBytesSec = false, hasItemsSec = false;
    int maxArguments = 0, maxTemplates = 0;
    QString time_unit;  // if same for all, otherwise "us" as default
};

//
// BenchResults
struct BenchResults {
    /*
     * Data
     */
    BenchMeta meta;
    BenchContext context;
    QVector<BenchData> benchmarks;
    
    
    /*
     * Static functions
     */
    // Replace argument and/or template with glyph in BenchData name
    static QString extractData(const BenchData &data, int argIdx, int tpltIdx, const QString &glyph = "",
                               int argIdx2 = -1, int tpltIdx2 = -1, const QString &glyph2 = "");
    // Replace argument with glyph in BenchData name
    static inline QString extractArgument(const BenchData &data, int argIdx, const QString &argGlyph)
    {
        return extractData(data, argIdx, -1, argGlyph);
    }
    // Replace template with glyph in BenchData name
    static inline QString extractTemplate(const BenchData &data, int tpltIdx, const QString &tpltGlyph)
    {
        return extractData(data, -1, tpltIdx, tpltGlyph);
    }
    // Try to extract special name+value from template name (JOMT specific)
    static QPair<double, QString> convertCustomDataSize(const QString &tplt);
    
    // Convert parameter name to value (check special names, use incremented fallback if all else fail)
    static double getParamValue(const QString &name, QString &custDataName,
                                bool &custDataAxis, double &fallbackIdx);
    
    /*
     * Member functions
     */
    // Ordered vector of all BenchData indexes
    QVector<int> segmentAll() const;
    
    // Each BenchData in its own subset
    QVector<BenchSubset> segmentEach() const;
    
    // Each Family in its own subset
    QVector<BenchSubset> segmentFamilies() const;
    
    // Each Family from index vector in its own subset
    QVector<BenchSubset> segmentFamilies(const QVector<int> &subset) const;
    
    // Each Container from index vector in its own subset
    QVector<BenchSubset> segmentContainers(const QVector<int> &subset) const;
    
    // Each BaseName in its own subset
    QVector<BenchSubset> segmentBaseNames() const;
    
    // Each BaseName from index vector in its own subset
    QVector<BenchSubset> segmentBaseNames(const QVector<int> &subset) const;
    
    // Each 'full name % param1 % param2' from index vector in its own subset
    QVector<BenchSubset> segment2DNames(const QVector<int> &subset,
                                        bool isArg1, int idx1, bool isArg2, int idx2) const;
    // Each Argument from index vector in its own subset
    QVector<BenchSubset> segmentArguments(const QVector<int> &subset, int argIdx) const;
    
    // Each Template from index vector in its own subset
    QVector<BenchSubset> segmentTemplates(const QVector<int> &subset, int tpltIdx) const;
    
    // Each Argument/Template from index vector in its own subset
    QVector<BenchSubset> segmentParam(bool isArgument, const QVector<int> &subset, int idx) const;
    
    //
    // Each Benchmark from vector in its own subset % Argument
    QVector<BenchSubset> groupArgument(const QVector<int> &subset,
                                       int argIdx, const QString &argGlyph) const;
    // Each Benchmark from vector in its own subset % Template
    QVector<BenchSubset> groupTemplate(const QVector<int> &subset,
                                       int tpltIdx, const QString &tpltGlyph) const;
    // Each Benchmark from vector in its own subset % Argument/Template
    QVector<BenchSubset> groupParam(bool isArgument, const QVector<int> &subset,
                                    int idx, const QString &glyph) const;
    
    // Get Benchmark full name
    QString getBenchName(int index) const;
    // Get Argument/Template name
    QString getParamName(bool isArgument, int benchIdx, int paramIdx) const;
    
    //
    // Merge results (rename BenchData if already exists)
    void appendResults(const BenchResults &bchRes);
    // Merge results (overwrite BenchData if already exists)
    void overwriteResults(const BenchResults &bchRes);
    
};


#endif // BENCHMARK_DATA_H
