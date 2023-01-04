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

#include <QMap>

#define BCHRES_DEBUG false
#include <QDebug>


/**************************************************************************************************
*
* Static functions
*
**************************************************************************************************/

QString BenchResults::extractData(const BenchData &data, int argIdx, int tpltIdx, const QString &glyph,
                           int argIdx2, int tpltIdx2, const QString &glyph2)
{
    int ttlArgs  = data.arguments.size();
    int ttlTplts = data.templates.size();
    
    // Check indexes validity
    if (argIdx+1 > ttlArgs) {
        if (BCHRES_DEBUG) qDebug() << "extractData(): no argument at index"
                                   << argIdx   << "of:" << data.run_name;
        return "";
    }
    if (argIdx2+1 > ttlArgs) {
        if (BCHRES_DEBUG) qDebug() << "extractData(): no argument at index"
                                   << argIdx2  << "of:" << data.run_name;
        return "";
    }
    if (tpltIdx+1 > ttlTplts) {
        if (BCHRES_DEBUG) qDebug() << "extractData(): no template at index"
                                   << tpltIdx  << "of:" << data.run_name;
        return "";
    }
    if (tpltIdx2+1 > ttlTplts) {
        if (BCHRES_DEBUG) qDebug() << "extractData(): no template at index"
                                   << tpltIdx2 << "of:" << data.run_name;
        return "";
    }
    
    QString sRes = data.base_name;
    bool hasBoth  = argIdx2 >= 0 || tpltIdx2 >= 0;
    bool lastArgs = ttlArgs >= 2
                && (argIdx  == ttlArgs-1 || argIdx  == ttlArgs-2)
                && (argIdx2 == ttlArgs-1 || argIdx2 == ttlArgs-2);
    
    // Templates
    QString sTemplates;
    for (int idx=0; idx<ttlTplts; ++idx)
    {
        if ( !sTemplates.isEmpty() ) sTemplates += ", ";
        if (idx == tpltIdx)
            sTemplates += glyph;
        else if (idx == tpltIdx2)
            sTemplates += glyph2;
        else
            sTemplates += data.templates[idx];
    }
    if ( !sTemplates.isEmpty() )
        sRes += "<" + sTemplates + ">";
    
    // Arguments
    for (int idx=0; idx<ttlArgs; ++idx)
    {
        if (idx != argIdx && idx != argIdx2)
            sRes += "/" + data.arguments[idx];
        else if (!lastArgs && (hasBoth || idx != ttlArgs-1)) // Not last
        {
            if (idx == argIdx)
                sRes += "/" + glyph;
            else
                sRes += "/" + glyph2;
        }
    }
    
    return sRes;
}

/**************************************************************************************************/

QPair<double, QString> BenchResults::convertCustomDataSize(const QString &tplt)
{
    QPair<double, QString> res(0., "");
    
    double mult = 0.;
    if      ( tplt.startsWith("data8")  ) mult = 1.;
    else if ( tplt.startsWith("data16") ) mult = 2.;
    else if ( tplt.startsWith("data32") ) mult = 4.;
    else if ( tplt.startsWith("data64") ) mult = 8.;
    
    if (mult > 0.)
    {
        int in  = tplt.indexOf('<');
        int out = tplt.lastIndexOf('>');
        
        if (in > 0 && out > in)
        {
            QString val = tplt.mid(in+1, out-in-1);
            bool ok = false;
            res.first = val.toDouble(&ok);
            if (ok) {
                res.first *= mult;
                res.second = "Data (bytes)";
            }
        }
    }
    
    return res;
}

/**************************************************************************************************/

double BenchResults::getParamValue(const QString &name, QString &custDataName, bool &custDataAxis, double &fallbackIdx)
{
    bool ok = false;
    double val = name.toDouble(&ok);
    if (!ok)
    {
        if (custDataAxis) {
            // Check if custom data size template
            const auto &custData = convertCustomDataSize(name);
            if ( !custData.second.isEmpty() ) {
                val = custData.first;
                if (custDataName.isEmpty())
                    custDataName = custData.second;
            }
            else {
                val = fallbackIdx++;
                custDataAxis = false;
            }
        }
        else
            val = fallbackIdx++;
    }
    else
        custDataAxis = false;
    
    return val;
}

/**************************************************************************************************
*
* Member functions
*
**************************************************************************************************/

QVector<int> BenchResults::segmentAll() const
{
    QVector<int> allRes;
    allRes.reserve( benchmarks.size() );
    
    for (int idx = 0; idx < benchmarks.size(); ++idx) {
        allRes.push_back(idx);
    }
    
    return allRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentEach() const
{
    QVector<BenchSubset> eachRes;
    eachRes.reserve( benchmarks.size() );
    
    for (int idx = 0; idx < benchmarks.size(); ++idx)
    {
        const BenchData &bchData = benchmarks[idx];
        eachRes.push_back( BenchSubset(bchData.name, idx) );
    }
    
    return eachRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentFamilies() const
{
    QVector<BenchSubset> famRes;
    QMap<QString, int> famMap;
    
    for (int idx = 0; idx < benchmarks.size(); ++idx)
    {
        const BenchData &bchData = benchmarks[idx];
                
        if ( !famMap.contains(bchData.family) )
        {
            famMap[bchData.family] = famRes.size();
            famRes.push_back( BenchSubset(bchData.family) );
        }
        // Append to associated entry
        famRes[famMap[bchData.family]].idxs.push_back(idx);
    }
    
    return famRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentFamilies(const QVector<int> &subset) const
{
    QVector<BenchSubset> famRes;
    QMap<QString, int> famMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        if ( !famMap.contains(bchData.family) )
        {
            famMap[bchData.family] = famRes.size();
            famRes.push_back( BenchSubset(bchData.family) );
        }
        // Append to associated entry
        famRes[famMap[bchData.family]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(famRes))
        if (BCHRES_DEBUG) qDebug() << "familySub:" << sub.name << "->" << sub.idxs;
    
    return famRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentContainers(const QVector<int> &subset) const
{
    QVector<BenchSubset> ctnRes;
    QMap<QString, int> ctnMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        if ( !ctnMap.contains(bchData.container) )
        {
            ctnMap[bchData.container] = ctnRes.size();
            ctnRes.push_back( BenchSubset(bchData.container) );
        }
        // Append to associated entry
        ctnRes[ctnMap[bchData.container]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(ctnRes))
        if (BCHRES_DEBUG) qDebug() << "containerSub:" << sub.name << "->" << sub.idxs;
    
    return ctnRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentBaseNames() const
{
    QVector<BenchSubset> nameRes;
    QMap<QString, int> nameMap;
    
    for (int idx = 0; idx < benchmarks.size(); ++idx)
    {
        const BenchData &bchData = benchmarks[idx];
                
        if ( !nameMap.contains(bchData.base_name) )
        {
            nameMap[bchData.base_name] = nameRes.size();
            nameRes.push_back( BenchSubset(bchData.base_name) );
        }
        // Append to associated entry
        nameRes[nameMap[bchData.base_name]].idxs.push_back(idx);
    }
    
    return nameRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentBaseNames(const QVector<int> &subset) const
{
    QVector<BenchSubset> nameRes;
    QMap<QString, int> nameMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        if ( !nameMap.contains(bchData.base_name) )
        {
            nameMap[bchData.base_name] = nameRes.size();
            nameRes.push_back( BenchSubset(bchData.base_name) );
        }
        // Append to associated entry
        nameRes[nameMap[bchData.base_name]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(nameRes))
        if (BCHRES_DEBUG) qDebug() << "nameSub:" << sub.name << "->" << sub.idxs;
    
    return nameRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segment2DNames(const QVector<int> &subset,
                                    bool isArg1, int idx1, bool isArg2, int idx2) const
{
    QVector<BenchSubset> nameRes;
    QMap<QString, int> nameMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        QString difName;
        if (isArg1) {
            if (isArg2)
                difName = extractData(bchData, idx1, -1, "X", idx2, -1, "Z");
            else
                difName = extractData(bchData, idx1, -1, "X", -1, idx2, "Z");
        }
        else {
            if (isArg2)
                difName = extractData(bchData, -1, idx1, "X", idx2, -1, "Z");
            else
                difName = extractData(bchData, -1, idx1, "X", -1, idx2, "Z");
        }
        
        if ( !nameMap.contains(difName) )
        {
            nameMap[difName] = nameRes.size();
            nameRes.push_back( BenchSubset(difName) );
        }
        // Append to associated entry
        nameRes[nameMap[difName]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(nameRes))
        if (BCHRES_DEBUG) qDebug() << "nameSub:" << sub.name << "->" << sub.idxs;
    
    return nameRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentArguments(const QVector<int> &subset, int argIdx) const
{
    QVector<BenchSubset> argRes;
    QMap<QString, int> argMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        if (bchData.arguments.size() <= argIdx) continue;
        
        const QString &param = bchData.arguments[argIdx];
        if ( !argMap.contains(param) )
        {
            argMap[param] = argRes.size();
            argRes.push_back( BenchSubset(param) ); //Full name but param
        }
        // Append to associated entry
        argRes[argMap[param]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(argRes))
        if (BCHRES_DEBUG) qDebug() << "argSub:" << sub.name << "->" << sub.idxs;
    
    return argRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentTemplates(const QVector<int> &subset, int tpltIdx) const
{
    QVector<BenchSubset> tpltRes;
    QMap<QString, int> tpltMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        if (bchData.templates.size() <= tpltIdx) continue;
        
        const QString &param = bchData.templates[tpltIdx];
        if ( !tpltMap.contains(param) )
        {
            tpltMap[param] = tpltRes.size();
            tpltRes.push_back( BenchSubset(param) );
        }
        // Append to associated entry
        tpltRes[tpltMap[param]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(tpltRes))
        if (BCHRES_DEBUG) qDebug() << "templateSub:" << sub.name << "->" << sub.idxs;
    
    return tpltRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::segmentParam(bool isArgument, const QVector<int> &subset, int idx) const
{
    if (isArgument)
        return segmentArguments(subset, idx);
    
    return segmentTemplates(subset, idx);
}

/**************************************************************************************************/
/**************************************************************************************************/

QVector<BenchSubset> BenchResults::groupArgument(const QVector<int> &subset,
                                                 int argIdx, const QString &argGlyph) const
{
    QVector<BenchSubset> argRes;
    QMap<QString, int> argMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists
        
        const BenchData &bchData = benchmarks[idx];
        const QString &bchID = extractArgument(bchData, argIdx, argGlyph);
        if ( bchID.isEmpty() )
            continue; //Ignore if incompatible
        
        if ( !argMap.contains(bchID) )
        {
            argMap[bchID] = argRes.size(); //Add ID to map
            argRes.push_back( BenchSubset(bchID) );
        }
        // Append to associated entry
        argRes[argMap[bchID]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(argRes))
        if (BCHRES_DEBUG) qDebug() << "argGSub:" << sub.name << "->" << sub.idxs;
    
    return argRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::groupTemplate(const QVector<int> &subset,
                                                 int tpltIdx, const QString &tpltGlyph) const
{
    QVector<BenchSubset> tpltRes;
    QMap<QString, int> tpltMap;
    
    for (int idx : subset)
    {
        if (idx >= benchmarks.size())
            continue; //No longer exists

        const BenchData &bchData = benchmarks[idx];
        const QString &bchID = extractTemplate(bchData, tpltIdx, tpltGlyph);
        if ( bchID.isEmpty() )
            continue; //Ignore if incompatible
        
        if ( !tpltMap.contains(bchID) )
        {
            tpltMap[bchID] = tpltRes.size(); //Add ID to map
            tpltRes.push_back( BenchSubset(bchID) );
        }
        // Append to associated entry
        tpltRes[tpltMap[bchID]].idxs.push_back(idx);
    }
    for (const BenchSubset& sub : qAsConst(tpltRes))
        if (BCHRES_DEBUG) qDebug() << "tpltGSub:" << sub.name << "->" << sub.idxs;
    
    return tpltRes;
}

/**************************************************************************************************/

QVector<BenchSubset> BenchResults::groupParam(bool isArgument, const QVector<int> &subset,
                                              int idx, const QString &glyph) const
{
    if (isArgument)
        return groupArgument(subset, idx, glyph);
    
    return groupTemplate(subset, idx, glyph);
}

/**************************************************************************************************/
/**************************************************************************************************/

QString BenchResults::getBenchName(int index) const
{
    Q_ASSERT(index >= 0 && index < benchmarks.size());
    return benchmarks[index].name;
}

QString BenchResults::getParamName(bool isArgument, int benchIdx, int paramIdx) const
{
    if (paramIdx < 0)
        return "";
    
    // Argument
    if (isArgument) {
        Q_ASSERT(benchmarks[benchIdx].arguments.size() > paramIdx);
        return benchmarks[benchIdx].arguments[paramIdx];
    }
    // Template
    Q_ASSERT(benchmarks[benchIdx].templates.size() > paramIdx);
    return benchmarks[benchIdx].templates[paramIdx];
}

/**************************************************************************************************/
/**************************************************************************************************/

void BenchResults::appendResults(const BenchResults &bchRes)
{
    // Benchmarks
    for (const auto& newBench : qAsConst(bchRes.benchmarks))
    {
        // Rename if needed
        QString tempName = newBench.name;
        int suffix = 1;
        
        int idx;
        do {
            idx = -1;
            for (int i=0; idx<0 && i<this->benchmarks.size(); ++i)
                if (this->benchmarks[i].name == tempName) idx = i;
            
            if (idx >= 0) {
                tempName = newBench.name;
                tempName.insert(newBench.base_name.size(), "_" + QString::number(++suffix));
            }
        } while (idx >= 0);
        
        // Apply and append
        if (newBench.name == tempName)
            this->benchmarks.append(newBench);
        else {
            BenchData cpyBench = newBench;
            cpyBench.name     = tempName;
            cpyBench.run_name = tempName;
            cpyBench.base_name += "_" + QString::number(suffix);
            
            this->benchmarks.append(cpyBench);
        }
        if (BCHRES_DEBUG) qDebug() << "newBench:" << newBench.name << "|" << tempName;
    }
    
    // Meta
    if (this->meta.maxArguments < bchRes.meta.maxArguments)
        this->meta.maxArguments = bchRes.meta.maxArguments;
    if (this->meta.maxTemplates < bchRes.meta.maxTemplates)
        this->meta.maxTemplates = bchRes.meta.maxTemplates;
    if (this->meta.time_unit != bchRes.meta.time_unit)
        this->meta.time_unit = "us";
    
    this->meta.hasAggregate  |= bchRes.meta.hasAggregate;
    this->meta.onlyAggregate &= bchRes.meta.onlyAggregate;
    this->meta.hasCv         |= bchRes.meta.hasCv;
    this->meta.hasBytesSec   |= bchRes.meta.hasBytesSec;
    this->meta.hasItemsSec   |= bchRes.meta.hasItemsSec;
}

/**************************************************************************************************/

void BenchResults::overwriteResults(const BenchResults &bchRes)
{
    // Benchmarks
    for (const auto& newBench : qAsConst(bchRes.benchmarks))
    {
        int idx = -1;
        for (int i=0; idx<0 && i<this->benchmarks.size(); ++i)
            if (this->benchmarks[i].name == newBench.name) idx = i;
        
        if (idx < 0)
            this->benchmarks.append(newBench);
        else
            this->benchmarks[idx] = newBench;
    }
    
    // Meta
    if (this->meta.maxArguments < bchRes.meta.maxArguments)
        this->meta.maxArguments = bchRes.meta.maxArguments;
    if (this->meta.maxTemplates < bchRes.meta.maxTemplates)
        this->meta.maxTemplates = bchRes.meta.maxTemplates;
    if (this->meta.time_unit != bchRes.meta.time_unit)
        this->meta.time_unit = "us";
    
    this->meta.hasAggregate  |= bchRes.meta.hasAggregate;
    this->meta.onlyAggregate &= bchRes.meta.onlyAggregate;
    this->meta.hasCv         |= bchRes.meta.hasCv;
    this->meta.hasBytesSec   |= bchRes.meta.hasBytesSec;
    this->meta.hasItemsSec   |= bchRes.meta.hasItemsSec;
}

/**************************************************************************************************/
