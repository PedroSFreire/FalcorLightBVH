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
#pragma once
#include "LightBVHTypes.slang"
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Scene/Lights/LightCollection.h"
#include "Utils/Math/AABB.h"
#include "Utils/Math/Vector.h"
#include "Utils/UI/Gui.h"
#include <functional>
#include <memory>
#include <vector>
#include <vector>

namespace Falcor
{
    class LightBVHBuilder;

    /** Utility class representing a light sampling BVH.

        This is binary BVH over all emissive triangles as described by Moreau and Clarberg,
        "Importance Sampling of Many Lights on the GPU", Ray Tracing Gems, Ch. 18, 2019.

        Before being used, the BVH needs to have been built using LightBVHBuilder::build().
        The data can be both used on the CPU (using traverseBVH() or on the GPU by:
          1. import LightBVH;
          2. Declare a variable of type LightBVH in your shader.
          3. Call setShaderData() to bind the BVH resources.

        TODO: Rename all things 'triangle' to 'light' as the BVH can be used for other light types.
    */
    class FALCOR_API LightBVH
    {
    public:
        using SharedPtr = std::shared_ptr<LightBVH>;
        using SharedConstPtr = std::shared_ptr<const LightBVH>;

        struct NodeLocation
        {
            uint32_t nodeIndex;
            uint32_t depth;

            NodeLocation() : nodeIndex(0), depth(0) {}
            NodeLocation(uint32_t _nodeIndex, uint32_t _depth) : nodeIndex(_nodeIndex), depth(_depth) {}
        };

        /** Function called on each node by traverseBVH().
            \param[in] location The location of the node in the tree.
            \return True if the traversal should continue, false otherwise.
        */
        using NodeFunction = std::function<bool(const NodeLocation& location)>;

        /** Creates an empty LightBVH object. Use a LightBVHBuilder to build the BVH.
            \param[in] pLightCollection The light collection around which the BVH will be built.
        */
        static SharedPtr create(const LightCollection::SharedConstPtr& pLightCollection);

        /** Refit all the BVH nodes to the underlying geometry, without changing the hierarchy.
            The BVH needs to have been built before trying to refit it.
            \param[in] pRenderContext The render context.
        */
        void refit(RenderContext* pRenderContext);
        void LightBVH::TLASrefit(RenderContext* pRenderContext);
        void LightBVH::BLASrefit(RenderContext* pRenderContext, uint32_t lightId);
        /** Perform a depth-first traversal of the BVH and run a function on each node.
            \param[in] evalInternal Function called on each internal node.
            \param[in] evalLeaf Function called on each leaf node.
            \param[in] rootNodeIndex The index of the node to start traversing.
        */

        void traverseTLAS(const NodeFunction& evalInternal, const NodeFunction& evalLeaf, uint32_t rootNodeIndex = 0);
        void traverseBLAS(const NodeFunction& evalInternal, const NodeFunction& evalLeaf, uint32_t lightId);


        
        struct BVHStats
        {
            std::vector<uint32_t> TLASNodeCountPerLevel;         ///< For each level in the tree, how many nodes are there.

            uint32_t TLASHeight = 0;                            ///< Number of edges on the longest path between the root node and a leaf.
            std::vector<uint32_t> BLASHeight;                            ///< Number of edges on the longest path between the root node and a leaf.
            uint32_t minTLASDepth = 0;                           ///< Number of edges on the shortest path between the root node and a leaf.
            std::vector<uint32_t> minBLASDepth;                           ///< Number of edges on the shortest path between the root node and a leaf.
            uint32_t TLASByteSize = 0;                           ///< Number of bytes occupied by the TLAS.
            uint32_t BLASByteSize = 0;                           ///< Number of bytes occupied by all BLASes.
            // seems useless uint32_t BLASByteSize = 0;                           ///< Number of bytes occupied by the BLASes.
            uint32_t TLASInternalNodeCount = 0;                  ///< Number of internal nodes inside the TLAS.
            std::vector<uint32_t> BLASInternalNodeCount;                  ///< Number of internal nodes inside each BLAS.
            uint32_t TLASLeafNodeCount = 0;                      ///< Number of leaf nodes inside the TLAS.
            uint32_t TLASTriangleCount = 0;                      ///< Number of triangles inside the TLAS.
            std::vector<uint32_t> BLASLeafNodeCount;                      ///< Number of leaf nodes inside each BLAS.
            std::vector<uint32_t> BLASTriangleCount;                      ///< Number of triangles inside each BLAS.
        };

        /** Returns stats.
        */
        const BVHStats& getStats() const { return mBVHStats; }

        /** Is the BVH valid.
            \return true if the BVH is ready for use.
        */
        virtual bool isValid() const { return mIsValid; }

        /** Render the UI. This default implementation just shows the stats.
        */
        virtual void renderUI(Gui::Widgets& widget);

        /** Bind the light BVH into a shader variable.
            \param[in] var The shader variable to set the data into.
        */
        virtual void setShaderData(ShaderVar const& var) const;

    protected:
        LightBVH(const LightCollection::SharedConstPtr& pLightCollection);

        void finalize();
        void computeStats();
        void computeTLASStats();
        void computeBLASStats(uint32_t lightId);
        void updateNodeIndices();
        void updateTLASIndices();
        void updateBLASIndices(uint32_t lightId);
        void renderStats(Gui::Widgets& widget, const BVHStats& stats) const;

        void uploadCPUBuffers(const std::vector<uint32_t>& triangleIndices, const std::vector<uint64_t>& triangleBitmasks, const std::vector<uint32_t>& lightIndices, const std::vector<uint64_t>& lightBitmasks);
        void syncDataToCPU() const;

        /** Invalidate the BVH.
        */
        virtual void clear();

        struct RefitEntryInfo
        {
            uint32_t offset = 0;    ///< Offset into the 'mpNodeIndicesBuffer' buffer.
            uint32_t count = 0;     ///< The number of nodes at each level.
        };

        // Internal state
        const LightCollection::SharedConstPtr mpLightCollection;

        ComputePass::SharedPtr                      mBLASLeafUpdater;                   ///< Compute pass for refitting internal nodes.
        ComputePass::SharedPtr                      mBLASInternalUpdater;               ///< Compute pass for refitting internal nodes.
        ComputePass::SharedPtr                      mTLASLeafUpdater;                   ///< Compute pass for refitting internal nodes.
        ComputePass::SharedPtr                      mTLASInternalUpdater;               ///< Compute pass for refitting internal nodes.
        // CPU resources

        mutable std::vector<PackedNode>             mTLAS;                              ///< CPU-side copy of packed TLAS nodes.
        mutable std::vector<PackedNode>             mBLAS;                              ///< CPU-side copy of packed BLASes nodes per light.
        mutable std::vector<uint32_t>               lightNodeIndices;                   ///< Array of first node indices of each light.
        mutable uint32_t                            mNumLights;                         //number of emissive meshes being considered

        std::vector<uint32_t>                       mTLASIndices;                       ///< Array of all node indices sorted by tree depth.
        std::vector<std::vector<uint32_t>>          mBLASIndices;                       ///< Array of all node indices sorted by tree depth.
        std::vector<std::vector<RefitEntryInfo>>    mPerDepthBLASRefitEntryInfo;        ///< Array containing for each level the number of internal nodes as well as the corresponding offset into 'mpNodeIndicesBuffer'; the very last entry contains the same data, but for all leaf nodes instead.
        std::vector<RefitEntryInfo>                 mPerDepthTLASRefitEntryInfo;        ///< Array containing for each level the number of internal nodes as well as the corresponding offset into 'mpNodeIndicesBuffer'; the very last entry contains the same data, but for all leaf nodes instead.
        uint32_t                                    mMaxTriangleCountPerLeaf = 0;       ///< After the BVH is built, this contains the maximum light count per leaf node.
        BVHStats                                    mBVHStats;
        bool                                        mIsValid = false;                   ///< True when the BVH has been built.
        mutable bool                                mIsCpuDataValid = false;            ///< Indicates whether the CPU-side data matches the GPU buffers.

        // GPU resources
        Buffer::SharedPtr                           mpBVHNodesBuffer;                   ///< Buffer holding all BVH nodes.
        Buffer::SharedPtr                           mpTLASNodesBuffer;                  ///< Buffer holding all BVH nodes.
        Buffer::SharedPtr                           mpBLASNodesBuffer;                  ///< Buffer holding all BVH nodes.
        Buffer::SharedPtr                           mpTriangleIndicesBuffer;            ///< Triangle indices sorted by leaf node. Each leaf node refers to a contiguous array of triangle indices.
        Buffer::SharedPtr                           mpTriangleBitmasksBuffer;           ///< Array containing the per triangle bit pattern retracing the tree traversal to reach the triangle: 0=left child, 1=right child.
        Buffer::SharedPtr                           mpLightIndicesBuffer;               ///< Light indices sorted by leaf node. Each leaf node refers to a contiguous array of light indices.
        Buffer::SharedPtr                           mpLightBitmasksBuffer;              ///< Array containing the per light bit pattern retracing the tree traversal to reach the triangle: 0=left child, 1=right child.
        Buffer::SharedPtr                           mpTLASIndicesBuffer;                ///< Buffer holding all node indices sorted by tree depth. This is used for BVH refit.
        std::vector<Buffer::SharedPtr>              mpBLASIndicesBuffer;                ///< Buffer holding all node indices sorted by tree depth. This is used for BVH refit.

        friend LightBVHBuilder;
    };
}
