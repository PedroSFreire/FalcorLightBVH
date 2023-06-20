/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "TwoLevelLightBVH.h"
#include "Core/Assert.h"
#include "Core/API/RenderContext.h"
#include "Utils/Timing/Profiler.h"

namespace
{
    const char kShaderFile[] = "Rendering/Lights/TwoLevelLightBVHRefit.cs.slang";
}

namespace Falcor
{
    TwoLevelLightBVH::SharedPtr TwoLevelLightBVH::create(const LightCollection::SharedConstPtr& pLightCollection)
    {
        return SharedPtr(new TwoLevelLightBVH(pLightCollection));
    }


    void TwoLevelLightBVH::TLASrefit(RenderContext* pRenderContext)
    {
        // Update all  TLAS leaf nodes.
        {
            auto var = mTLASLeafUpdater->getVars()["CB"];
            mpLightCollection->setShaderData(var["gLights"]);
            setShaderData(var["gLightBVH"]);
            var["gNodeIndices"] = mpTLASIndicesBuffer;

            const uint32_t nodeCount = mPerDepthTLASRefitEntryInfo.back().count;
            FALCOR_ASSERT(nodeCount > 0);
            var["gFirstNodeOffset"] = mPerDepthTLASRefitEntryInfo.back().offset;
            var["gNodeCount"] = nodeCount;

            mTLASLeafUpdater->execute(pRenderContext, nodeCount, 1, 1);
        }

        // Update all TLAS internal nodes.
        {
            auto var = mTLASInternalUpdater->getVars()["CB"];
            mpLightCollection->setShaderData(var["gLights"]);
            setShaderData(var["gLightBVH"]);
            var["gNodeIndices"] = mpTLASIndicesBuffer;

            // Note that mBVHStats.treeHeight may be 0, in which case there is a single leaf and no internal nodes.
            for (int depth = (int)mBVHStats.TLASHeight - 1; depth >= 0; --depth)
            {
                const uint32_t nodeCount = mPerDepthTLASRefitEntryInfo[depth].count;
                FALCOR_ASSERT(nodeCount > 0);
                var["gFirstNodeOffset"] = mPerDepthTLASRefitEntryInfo[depth].offset;
                var["gNodeCount"] = nodeCount;

                mTLASInternalUpdater->execute(pRenderContext, nodeCount, 1, 1);
            }
        }

    }


    void TwoLevelLightBVH::BLASrefit(RenderContext* pRenderContext, uint32_t lightId)
    {   
        // Update all BLAS leaf nodes.
        {
            auto var = mBLASLeafUpdater->getVars()["CB"];
            mpLightCollection->setShaderData(var["gLights"]);
            setShaderData(var["gLightBVH"]);
            var["gNodeIndices"] = mpBLASIndicesBuffer[lightId];

            const uint32_t nodeCount = mPerDepthBLASRefitEntryInfo[lightId].back().count;
            FALCOR_ASSERT(nodeCount > 0);
            var["gFirstNodeOffset"] = mPerDepthBLASRefitEntryInfo[lightId].back().offset;
            var["gNodeCount"] = nodeCount;

            mBLASLeafUpdater->execute(pRenderContext, nodeCount, 1, 1);
        }

        // Update all BLAS internal nodes.
        {
            auto var = mBLASInternalUpdater->getVars()["CB"];
            mpLightCollection->setShaderData(var["gLights"]);
            setShaderData(var["gLightBVH"]);
            var["gNodeIndices"] = mpBLASIndicesBuffer[lightId];

            // Note that mBVHStats.treeHeight may be 0, in which case there is a single leaf and no internal nodes.
            for (int depth = (int)mBVHStats.BLASHeight[lightId] - 1; depth >= 0; --depth)
            {
                const uint32_t nodeCount = mPerDepthBLASRefitEntryInfo[lightId][depth].count;
                FALCOR_ASSERT(nodeCount > 0);
                var["gFirstNodeOffset"] = mPerDepthBLASRefitEntryInfo[lightId][depth].offset;
                var["gNodeCount"] = nodeCount;

                mBLASInternalUpdater->execute(pRenderContext, nodeCount, 1, 1);
            }
        }
    }



   
    bool TwoLevelLightBVH::refit(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE("LightBVH::refit()");

        FALCOR_ASSERT(mIsValid);
        bool updated = false;
        //catch rebuild thread
        if (threadOn) {
            threadOn = false;
            rebuildThread.join();
        }
        using namespace std::literals::chrono_literals;
        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < mpLightCollection->changedLights.size(); i++) {
            BLASrefit(pRenderContext, mpLightCollection->changedLights[i]);
            updated = true;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << duration.count() << std::endl;
        printf("number of refoted lights = %d", (int)mpLightCollection->changedLights.size());
        if (updated) {       
            TLASrefit(pRenderContext);
            mIsCpuDataValid = false;
            return true;
        }
            
        return false;
        
    }

    void TwoLevelLightBVH::renderUI(Gui::Widgets& widget)
    {
        // Render the BVH stats.
        //todo redo
        renderStats(widget, getStats());
    }

    void TwoLevelLightBVH::renderStats(Gui::Widgets& widget, const BVHStats& stats) const
    {
        const std::string statsStr =
            "  TLAS height:                 " + std::to_string(stats.TLASHeight) + "\n" +
            "  TLAS Min depth:              " + std::to_string(stats.minTLASDepth) + "\n" +
            "  TLAS Size:                   " + std::to_string(stats.TLASByteSize) + " bytes\n" +
            "  BLAS Collective Size:        " + std::to_string(stats.BLASByteSize) + " bytes\n" +
            "  TLAS Internal node count:    " + std::to_string(stats.TLASInternalNodeCount) + "\n" +
            "  TLAS Leaf node count:        " + std::to_string(stats.TLASLeafNodeCount) + "\n" +
            "  TLAS Triangle count:         " + std::to_string(stats.TLASTriangleCount) + "\n";
        widget.text(statsStr);

        if (auto nodeGroup = widget.group("TLAS count per level"))
        {
            std::string countStr;
            for (uint32_t level = 0; level < stats.TLASNodeCountPerLevel.size(); ++level)
            {
                countStr += "  Node count at level " + std::to_string(level) + ": " + std::to_string(stats.TLASNodeCountPerLevel[level]) + "\n";
            }
            if (!countStr.empty()) countStr.pop_back();
            nodeGroup.text(countStr);
        }

    }

    void TwoLevelLightBVH::clear()
    {
        // Reset all CPU data.
        mTLAS.clear();
        mBLAS.clear();
        mPerDepthTLASRefitEntryInfo.clear();
        mPerDepthBLASRefitEntryInfo.clear();
        mMaxTriangleCountPerLeaf = 0;
        mBVHStats = BVHStats();
        mIsValid = false;
        mIsCpuDataValid = false;
    }

    TwoLevelLightBVH::TwoLevelLightBVH(const LightCollection::SharedConstPtr& pLightCollection) : mpLightCollection(pLightCollection)
    {
      
        mBLASLeafUpdater = ComputePass::create(kShaderFile, "updateBLASLeafNodes");
        mBLASInternalUpdater = ComputePass::create(kShaderFile, "updateBLASInternalNodes");
        mTLASLeafUpdater = ComputePass::create(kShaderFile, "updateTLASLeafNodes");
        mTLASInternalUpdater = ComputePass::create(kShaderFile, "updateTLASInternalNodes"); 
    }


    void TwoLevelLightBVH::traverseTLAS(const NodeFunction& evalInternal, const NodeFunction& evalLeaf, uint32_t rootNodeIndex)
    {
        std::stack<NodeLocation> stack({NodeLocation{rootNodeIndex, 0}});
        while (!stack.empty())
        {
            const NodeLocation location = stack.top();
            stack.pop();

            if (mTLAS[location.nodeIndex].isLeaf())
            {
                if (!evalLeaf(location)) break;
            }
            else
            {
                if (!evalInternal(location)) break;

                // Push the children nodes onto the stack.
                auto node = mTLAS[location.nodeIndex].getInternalNode();
                stack.push(NodeLocation{ location.nodeIndex + 1, location.depth + 1 });
                stack.push(NodeLocation{ node.rightChildIdx, location.depth + 1 });
            }
        }
    }


    void TwoLevelLightBVH::traverseBLAS(const NodeFunction& evalInternal, const NodeFunction& evalLeaf, uint32_t lightId)
    {
        std::stack<NodeLocation> stack({ NodeLocation{lightNodeIndices[lightId], 0} });
        while (!stack.empty())
        {
            const NodeLocation location = stack.top();
            stack.pop();

            if (mBLAS[location.nodeIndex].isLeaf())
            {
                if (!evalLeaf(location)) break;
            }
            else
            {
                if (!evalInternal(location)) break;

                // Push the children nodes onto the stack.
                auto node = mBLAS[location.nodeIndex].getInternalNode();
                stack.push(NodeLocation{ location.nodeIndex + 1, location.depth + 1 });
                stack.push(NodeLocation{ node.rightChildIdx, location.depth + 1 });
            }
        }
    }

    void TwoLevelLightBVH::finalize()
    {
        // This function is called after BVHs build has finished.
        computeStats();
        updateNodeIndices();
    }

    void TwoLevelLightBVH::computeStats()
    {
        mBVHStats.BLASHeight.clear();
        mBVHStats.BLASHeight.resize(mNumLights);
        mBVHStats.minBLASDepth.clear();
        mBVHStats.minBLASDepth.resize(mNumLights);
        mBVHStats.BLASInternalNodeCount.clear();
        mBVHStats.BLASInternalNodeCount.resize(mNumLights);
        mBVHStats.BLASLeafNodeCount.clear();
        mBVHStats.BLASLeafNodeCount.resize(mNumLights);
        mBVHStats.BLASTriangleCount.clear();
        mBVHStats.BLASTriangleCount.resize(mNumLights);

        computeTLASStats();
        for (uint32_t i = 0; i < mNumLights; i++) {
            computeBLASStats(i);
        }
        mBVHStats.BLASByteSize += (uint32_t)(mBLAS.size() * sizeof(mBLAS[0]));
        
    }

    void TwoLevelLightBVH::computeTLASStats()
    {
        FALCOR_ASSERT(isValid());
        mBVHStats.TLASNodeCountPerLevel.clear();
        mBVHStats.TLASNodeCountPerLevel.reserve(32);

        FALCOR_ASSERT(mMaxTriangleCountPerLeaf > 0);


        mBVHStats.TLASHeight = 0;
        mBVHStats.minTLASDepth = std::numeric_limits<uint32_t>::max();
        mBVHStats.TLASInternalNodeCount = 0;
        mBVHStats.TLASLeafNodeCount = 0;
        mBVHStats.TLASTriangleCount = 0;

        auto evalInternal = [&](const NodeLocation& location)
        {
 
            if (mBVHStats.TLASNodeCountPerLevel.size() <= location.depth) mBVHStats.TLASNodeCountPerLevel.push_back(1);
            else ++mBVHStats.TLASNodeCountPerLevel[location.depth];
            ++mBVHStats.TLASInternalNodeCount;
            return true;
        };
        auto evalLeaf = [&](const NodeLocation& location)
        {

            const auto node = mTLAS[location.nodeIndex].getLeafNode();

            if (mBVHStats.TLASNodeCountPerLevel.size() <= location.depth) mBVHStats.TLASNodeCountPerLevel.push_back(1);
            else ++mBVHStats.TLASNodeCountPerLevel[location.depth];

            ++mBVHStats.TLASLeafNodeCount;

            mBVHStats.TLASHeight = std::max(mBVHStats.TLASHeight, location.depth);
            mBVHStats.minTLASDepth = std::min(mBVHStats.minTLASDepth, location.depth);
            mBVHStats.TLASTriangleCount += node.triangleCount;
            return true;
        };
        traverseTLAS(evalInternal, evalLeaf);

        mBVHStats.TLASByteSize = (uint32_t)(mTLAS.size() * sizeof(mTLAS[0]));
    }

    void TwoLevelLightBVH::computeBLASStats(uint32_t lightId)
    {


        mBVHStats.BLASHeight[lightId] = 0;
        mBVHStats.minBLASDepth[lightId] = std::numeric_limits<uint32_t>::max();
        mBVHStats.BLASInternalNodeCount[lightId] = 0;
        mBVHStats.BLASLeafNodeCount[lightId] = 0;
        mBVHStats.BLASTriangleCount[lightId] = 0;

        auto evalInternal = [&](const NodeLocation& location)
        {

            ++mBVHStats.BLASInternalNodeCount[lightId];
            return true;
        };
        auto evalLeaf = [&](const NodeLocation& location)
        {
            const auto node = mBLAS[location.nodeIndex].getLeafNode();


            ++mBVHStats.BLASLeafNodeCount[lightId];

            mBVHStats.BLASHeight[lightId] = std::max(mBVHStats.BLASHeight[lightId], location.depth);
            mBVHStats.minBLASDepth[lightId] = std::min(mBVHStats.minBLASDepth[lightId], location.depth);
            mBVHStats.BLASTriangleCount[lightId] += node.triangleCount;
            return true;
        };
        traverseBLAS(evalInternal, evalLeaf, lightId);
    }


    void TwoLevelLightBVH::updateTLASIndices()
    {
        // The nodes of the TLAS are stored in depth-first order. To simplify the work of the refit kernels,
        // they are first run on all leaf nodes, and then on all internal nodes on a per level basis.
        // In order to do that, we need to compute how many internal nodes are stored at each level.
        FALCOR_ASSERT(isValid());
        mPerDepthTLASRefitEntryInfo.clear();
        mPerDepthTLASRefitEntryInfo.resize(mBVHStats.TLASHeight + 1);
        mPerDepthTLASRefitEntryInfo.back().count = mBVHStats.TLASLeafNodeCount;

        traverseTLAS(
            [&](const NodeLocation& location) { ++mPerDepthTLASRefitEntryInfo[location.depth].count; return true; },
            [](const NodeLocation& location) { return true; }
        );

        std::vector<uint32_t> perDepthOffset(mPerDepthTLASRefitEntryInfo.size(), 0);
        for (std::size_t i = 1; i < mPerDepthTLASRefitEntryInfo.size(); ++i)
        {
            uint32_t currentOffset = mPerDepthTLASRefitEntryInfo[i - 1].offset + mPerDepthTLASRefitEntryInfo[i - 1].count;
            perDepthOffset[i] = mPerDepthTLASRefitEntryInfo[i].offset = currentOffset;
        }

        // For validation purposes
        {
            uint32_t currentOffset = 0;
            for (const RefitEntryInfo& info : mPerDepthTLASRefitEntryInfo)
            {
                FALCOR_ASSERT(info.offset == currentOffset);
                currentOffset += info.count;
            }
            FALCOR_ASSERT(currentOffset == (mBVHStats.TLASInternalNodeCount + mBVHStats.TLASLeafNodeCount));
        }

        // Now that we know how many nodes are stored per level (excluding leaf nodes) and how many leaf nodes there are,
        // we can fill in the buffer with all the node indices sorted by tree level. The indices are stored as follows
        // <-- Indices to all internal nodes at level 0 --> | ... | <-- Indices to all internal nodes at level (treeHeight - 1) --> | <-- Indices to all leaf nodes -->
        mTLASIndices.clear();
        mTLASIndices.resize(mBVHStats.TLASInternalNodeCount + mBVHStats.TLASLeafNodeCount, 0);

        traverseTLAS(
            [&](const NodeLocation& location) { mTLASIndices[perDepthOffset[location.depth]++] = location.nodeIndex; return true; },
            [&](const NodeLocation& location) { mTLASIndices[perDepthOffset.back()++] = location.nodeIndex; return true; }
        );

        if (!mpTLASIndicesBuffer || mpTLASIndicesBuffer->getElementCount() < mTLASIndices.size())
        {
            mpTLASIndicesBuffer = Buffer::createStructured(sizeof(uint32_t), (uint32_t)mTLASIndices.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpTLASIndicesBuffer->setName("LightBVH::mpTLASIndicesBuffer");
        }

        mpTLASIndicesBuffer->setBlob(mTLASIndices.data(), 0, mTLASIndices.size() * sizeof(uint32_t));
    }



    void TwoLevelLightBVH::updateBLASIndices(uint32_t lightId)
    {
        // The nodes of the TLAS are stored in depth-first order. To simplify the work of the refit kernels,
        // they are first run on all leaf nodes, and then on all internal nodes on a per level basis.
        // In order to do that, we need to compute how many internal nodes are stored at each level.
        FALCOR_ASSERT(isValid());
        mPerDepthBLASRefitEntryInfo[lightId].clear();
        mPerDepthBLASRefitEntryInfo[lightId].resize(mBVHStats.BLASHeight[lightId] + 1);
        mPerDepthBLASRefitEntryInfo[lightId].back().count = mBVHStats.BLASLeafNodeCount[lightId];

        traverseBLAS(
            [&](const NodeLocation& location) { ++mPerDepthBLASRefitEntryInfo[lightId][location.depth].count; return true; },
            [](const NodeLocation& location) { return true; },lightId
        );

        std::vector<uint32_t> perDepthOffset(mPerDepthBLASRefitEntryInfo[lightId].size(), 0);
        for (std::size_t i = 1; i < mPerDepthBLASRefitEntryInfo[lightId].size(); ++i)
        {
            uint32_t currentOffset = mPerDepthBLASRefitEntryInfo[lightId][i - 1].offset + mPerDepthBLASRefitEntryInfo[lightId][i - 1].count;
            perDepthOffset[i] = mPerDepthBLASRefitEntryInfo[lightId][i].offset = currentOffset;
        }

        // For validation purposes
        {
            uint32_t currentOffset = 0;
            for (const RefitEntryInfo& info : mPerDepthBLASRefitEntryInfo[lightId])
            {
                FALCOR_ASSERT(info.offset == currentOffset);
                currentOffset += info.count;
            }
            FALCOR_ASSERT(currentOffset == (mBVHStats.BLASInternalNodeCount[lightId] + mBVHStats.BLASLeafNodeCount[lightId]));
        }

        // Now that we know how many nodes are stored per level (excluding leaf nodes) and how many leaf nodes there are,
        // we can fill in the buffer with all the node indices sorted by tree level. The indices are stored as follows
        // <-- Indices to all internal nodes at level 0 --> | ... | <-- Indices to all internal nodes at level (treeHeight - 1) --> | <-- Indices to all leaf nodes -->
        mBLASIndices[lightId].clear();
        mBLASIndices[lightId].resize(mBVHStats.BLASInternalNodeCount[lightId] + mBVHStats.BLASLeafNodeCount[lightId], 0);
        
        traverseBLAS(
            [&](const NodeLocation& location) { mBLASIndices[lightId][perDepthOffset[location.depth]++] = location.nodeIndex; return true; },
            [&](const NodeLocation& location) { mBLASIndices[lightId][perDepthOffset.back()++] = location.nodeIndex; return true; },lightId
        );

        if (!mpBLASIndicesBuffer[lightId] || mpBLASIndicesBuffer[lightId]->getElementCount() < mBLASIndices[lightId].size())
        {
            mpBLASIndicesBuffer[lightId] = Buffer::createStructured(sizeof(uint32_t), (uint32_t)mBLASIndices[lightId].size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpBLASIndicesBuffer[lightId]->setName("LightBVH::mpBLASIndicesBuffer");
        }

        mpBLASIndicesBuffer[lightId]->setBlob(mBLASIndices[lightId].data(), 0, mBLASIndices[lightId].size() * sizeof(uint32_t));
    }



    void TwoLevelLightBVH::updateNodeIndices()
    {

        mPerDepthBLASRefitEntryInfo.clear();
        mPerDepthBLASRefitEntryInfo.resize(mNumLights);

        mBLASIndices.clear();
        mBLASIndices.resize(mNumLights);

        mpBLASIndicesBuffer.clear();
        mpBLASIndicesBuffer.resize(mNumLights);

        updateTLASIndices();
        mBLASIndices2.clear();
        mBLASIndices2.resize(mBLAS.size());

        for(uint32_t i = 0; i < mNumLights; i++) {
            updateBLASIndices(i);
        }
    }

    void TwoLevelLightBVH::uploadCPUBuffers(const std::vector<uint32_t>& triangleIndices, const std::vector<uint64_t>& triangleBitmasks, const std::vector<uint32_t>& lightIndices, const std::vector<uint64_t>& lightBitmasks)
    {
        // Reallocate buffers if size requirements have changed.
        auto var = mBLASLeafUpdater->getRootVar()["CB"]["gLightBVH"];
               ///< Buffer containing all light id to first triangle BLAS index.
        if (!mpLightNodeIndicesBuffer || mpLightNodeIndicesBuffer->getElementCount() < lightIndices.size())
        {
            mpLightNodeIndicesBuffer = Buffer::createStructured(var["lightNodeIndices"], (uint32_t)lightNodeIndices.size(), Resource::BindFlags::None, Buffer::CpuAccess::Read, nullptr, false);
            mpLightNodeIndicesBuffer->setName("LightBVH::mpLightNodeIndicesBuffer");
        }
        if (!mpBLASChangedRootsBuffer || mpBLASChangedRootsBuffer->getElementCount() < lightIndices.size())
        {
            mpBLASChangedRootsBuffer = Buffer::createStructured(var["BLASChangedRoots"], (uint32_t)lightIndices.size(), Resource::BindFlags::None, Buffer::CpuAccess::Read, nullptr, false);
            mpBLASChangedRootsBuffer->setName("LightBVH::mpBLASChangedRootsBuffer");
        }
        if (!mpTLASNodesBuffer || mpTLASNodesBuffer->getElementCount() < mTLAS.size())
        {
            mpTLASNodesBuffer = Buffer::createStructured(var["TLAS"], (uint32_t)mTLAS.size(), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpTLASNodesBuffer->setName("LightBVH::mpTLASNodesBuffer");
        }
        if (!mpBLASNodesBuffer || mpBLASNodesBuffer->getElementCount() < mBLAS.size())
        {
            mpBLASNodesBuffer = Buffer::createStructured(var["BLAS"], (uint32_t)mBLAS.size(), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpBLASNodesBuffer->setName("LightBVH::mpBLASNodesBuffer");
        }
        if (!mpTriangleIndicesBuffer || mpTriangleIndicesBuffer->getElementCount() < triangleIndices.size())
        {
            mpTriangleIndicesBuffer = Buffer::createStructured(var["triangleIndices"], (uint32_t)triangleIndices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpTriangleIndicesBuffer->setName("LightBVH::mpTriangleIndicesBuffer");
        }
        if (!mpTriangleBitmasksBuffer || mpTriangleBitmasksBuffer->getElementCount() < triangleBitmasks.size())
        {
            mpTriangleBitmasksBuffer = Buffer::createStructured(var["triangleBitmasks"], (uint32_t)triangleBitmasks.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpTriangleBitmasksBuffer->setName("LightBVH::mpTriangleBitmasksBuffer");
        }


        if (!mpLightIndicesBuffer || mpLightIndicesBuffer->getElementCount() < lightIndices.size())
        {
            mpLightIndicesBuffer = Buffer::createStructured(var["lightIndices"], (uint32_t)lightIndices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpLightIndicesBuffer->setName("LightBVH::mpLightIndicesBuffer");
        }
        if (!mpLightBitmasksBuffer || mpLightBitmasksBuffer->getElementCount() < lightBitmasks.size())
        {
            mpLightBitmasksBuffer = Buffer::createStructured(var["lightBitmasks"], (uint32_t)lightBitmasks.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpLightBitmasksBuffer->setName("LightBVH::mpLightBitmasksBuffer");
        }



        FALCOR_ASSERT(mpLightNodeIndicesBuffer->getSize() >= lightNodeIndices.size() * sizeof(lightNodeIndices[0]));
        mpLightNodeIndicesBuffer->setBlob(lightNodeIndices.data(), 0, lightNodeIndices.size() * sizeof(lightNodeIndices[0]));

        FALCOR_ASSERT(mpTLASNodesBuffer->getElementCount() >= mTLAS.size());
        FALCOR_ASSERT(mpTLASNodesBuffer->getStructSize() == sizeof(mTLAS[0]));
        mpTLASNodesBuffer->setBlob(mTLAS.data(), 0, mTLAS.size() * sizeof(mTLAS[0]));

        FALCOR_ASSERT(mpBLASNodesBuffer->getElementCount() >= mBLAS.size());
        FALCOR_ASSERT(mpBLASNodesBuffer->getStructSize() == sizeof(mBLAS[0]));
        mpBLASNodesBuffer->setBlob(mBLAS.data(), 0, mBLAS.size() * sizeof(mBLAS[0]));

        FALCOR_ASSERT(mpTriangleIndicesBuffer->getSize() >= triangleIndices.size() * sizeof(triangleIndices[0]));
        mpTriangleIndicesBuffer->setBlob(triangleIndices.data(), 0, triangleIndices.size() * sizeof(triangleIndices[0]));

        FALCOR_ASSERT(mpTriangleBitmasksBuffer->getSize() >= triangleBitmasks.size() * sizeof(triangleBitmasks[0]));
        mpTriangleBitmasksBuffer->setBlob(triangleBitmasks.data(), 0, triangleBitmasks.size() * sizeof(triangleBitmasks[0]));

        FALCOR_ASSERT(mpLightIndicesBuffer->getSize() >= lightIndices.size() * sizeof(lightIndices[0]));
        mpTriangleIndicesBuffer->setBlob(lightIndices.data(), 0, lightIndices.size() * sizeof(lightIndices[0]));

        FALCOR_ASSERT(mpLightBitmasksBuffer->getSize() >= lightBitmasks.size() * sizeof(lightBitmasks[0]));
        mpTriangleBitmasksBuffer->setBlob(lightBitmasks.data(), 0, lightBitmasks.size() * sizeof(lightBitmasks[0]));

        mIsCpuDataValid = true;
    }


    void TwoLevelLightBVH::uploadTLASBuffer( const std::vector<uint32_t>& lightIndices, const std::vector<uint64_t>& lightBitmasks)
    {
        // Reallocate buffers if size requirements have changed.
        auto var = mBLASLeafUpdater->getRootVar()["CB"]["gLightBVH"];
        ///< Buffer containing all light id to first triangle BLAS index.
        if (!mpLightNodeIndicesBuffer || mpLightNodeIndicesBuffer->getElementCount() < lightIndices.size())
        {
            mpLightNodeIndicesBuffer = Buffer::createStructured(var["lightNodeIndices"], (uint32_t)lightNodeIndices.size(), Resource::BindFlags::None, Buffer::CpuAccess::Read, nullptr, false);
            mpLightNodeIndicesBuffer->setName("LightBVH::mpLightNodeIndicesBuffer");
        }
        
        if (!mpTLASNodesBuffer || mpTLASNodesBuffer->getElementCount() < mTLAS.size())
        {
            mpTLASNodesBuffer = Buffer::createStructured(var["TLAS"], (uint32_t)mTLAS.size(), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpTLASNodesBuffer->setName("LightBVH::mpTLASNodesBuffer");
        }


        
        FALCOR_ASSERT(mpTLASNodesBuffer->getElementCount() >= mTLAS.size());
        FALCOR_ASSERT(mpTLASNodesBuffer->getStructSize() == sizeof(mTLAS[0]));
        mpTLASNodesBuffer->setBlob(mTLAS.data(), 0, mTLAS.size() * sizeof(mTLAS[0]));

        FALCOR_ASSERT(mpLightIndicesBuffer->getSize() >= lightIndices.size() * sizeof(lightIndices[0]));
        mpTriangleIndicesBuffer->setBlob(lightIndices.data(), 0, lightIndices.size() * sizeof(lightIndices[0]));

        FALCOR_ASSERT(mpLightBitmasksBuffer->getSize() >= lightBitmasks.size() * sizeof(lightBitmasks[0]));
        mpTriangleBitmasksBuffer->setBlob(lightBitmasks.data(), 0, lightBitmasks.size() * sizeof(lightBitmasks[0]));


        mIsCpuDataValid = true;
    }

    void TwoLevelLightBVH::syncDataToCPU() const
    {
        if (!mIsValid || mIsCpuDataValid) return;

        // TODO: This is slow because of the flush. We should copy to a staging buffer
        // after the data is updated on the GPU and map the staging buffer here instead.
        //const void* const ptr = mpBVHNodesBuffer->map(Buffer::MapType::Read);
        //FALCOR_ASSERT(mNodes.size() > 0 && mNodes.size() <= mpBVHNodesBuffer->getElementCount());
        //std::memcpy(mNodes.data(), ptr, mNodes.size() * sizeof(mNodes[0]));
        //mpBVHNodesBuffer->unmap();

        const void* const ptrTLAS = mpTLASNodesBuffer->map(Buffer::MapType::Read);
        FALCOR_ASSERT(mTLAS.size() > 0 && mTLAS.size() <= mpTLASNodesBuffer->getElementCount());
        std::memcpy(mTLAS.data(), ptrTLAS, mTLAS.size() * sizeof(mTLAS[0]));
        mpTLASNodesBuffer->unmap();

        const void* const ptrBLAS = mpBLASNodesBuffer->map(Buffer::MapType::Read);
        FALCOR_ASSERT(mBLAS.size() > 0 && mBLAS.size() <= mpBLASNodesBuffer->getElementCount());
        std::memcpy(mBLAS.data(), ptrBLAS, mBLAS.size() * sizeof(mBLAS[0]));
        mpBLASNodesBuffer->unmap();
        mIsCpuDataValid = true;
    }



    void TwoLevelLightBVH::syncBLASDataToCPU() const
    {
        if (!mIsValid || mIsCpuDataValid) return;


        const void* const ptrBLAS = mpBLASChangedRootsBuffer->map(Buffer::MapType::Read);
        FALCOR_ASSERT(ChangedLights.size() > 0 && ChangedLights.size() <= mpBLASChangedRootsBuffer->getElementCount());
        std::memcpy(ChangedLights.data(), ptrBLAS, ChangedLights.size() * sizeof(ChangedLights[0]));
        mpBLASChangedRootsBuffer->unmap();



        mIsCpuDataValid = true;
    }

    void TwoLevelLightBVH::setShaderData(const ShaderVar& var) const
    {
        if (isValid())
        {
            FALCOR_ASSERT(var.isValid());
            var["TLAS"] = mpTLASNodesBuffer;
            var["BLASChangedRoots"] = mpBLASChangedRootsBuffer;
            var["lightNodeIndices"] = mpLightNodeIndicesBuffer;
            var["BLAS"] = mpBLASNodesBuffer;
            var["triangleIndices"] = mpTriangleIndicesBuffer;
            var["triangleBitmasks"] = mpTriangleBitmasksBuffer;
            var["lightIndices"] = mpLightIndicesBuffer;
            var["lightBitmasks"] = mpLightBitmasksBuffer;
        }
    }
}
