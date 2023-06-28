/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Math/MathConstants.slangh"

import Scene.Lights.LightCollection;
import Rendering.Lights.TwoLevelLightBVH;

cbuffer CB
{
    LightCollection         gLights;            ///< The light sources.
    RWTwoLevelLightBVH              gLightBVH;          ///< The BVH around the light sources.
    StructuredBuffer<uint>  gNodeIndices;       ///< Buffer containing the indices of all the nodes. The indices are sorted by depths and laid out contiguously in memory; the indices for all the leaves are placed in the lowest level.
    uint                    gFirstNodeOffset;   ///< The offset of the first node index in 'gNodeIndices' to be processed.
    uint                    gNodeCount;         ///< Amount of nodes that need to be processed.
    StructuredBuffer<uint>  gFirstNodeOffsetVec;   ///< The offset of the first node index in 'gNodeIndices' to be processed.
    StructuredBuffer<uint>  gNodeCountVec;         ///< Amount of nodes that need to be processed.
};






/** Compute shader for refitting the BLAS leaf nodes.
    The code assumes a leaf stores an indexed list of emissive triangles.
*/

[numthreads(256, 1, 1)]
void updateBLASLeafNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;
    uint count = DTid.x;
    uint i = 0;
    for(i=0; i<gLights.meshCount ; i++){
        if(gNodeCountVec[i] == 0)continue;
        if(count > gNodeCountVec[i])
            count -= gNodeCountVec[i];
        else
            break;
    }
    uint nodeIndex = gNodeIndices[gFirstNodeOffsetVec[i] + count];
    LeafNode node = gLightBVH.getBLASLeafNode(nodeIndex);

    // Update the node bounding box.
    float3 aabbMin = float3(FLT_MAX);
    float3 aabbMax = float3(-FLT_MAX);
    float3 normalsSum = float3(0.0f);

    for (uint i = 0; i < node.triangleCount; i++)
    {
        uint triangleIndex = gLightBVH.getNodeTriangleIndex(node, i);
        EmissiveTriangle tri = gLights.getTriangle(triangleIndex);
        for (uint vertexIndex = 0u; vertexIndex < 3u; ++vertexIndex)
        {
            aabbMin = min(aabbMin, tri.posW[vertexIndex]);
            aabbMax = max(aabbMax, tri.posW[vertexIndex]);
        }
        normalsSum += tri.normal;
    }

    node.attribs.setAABB(aabbMin, aabbMax);

    // Update the normal bounding cone.
    float coneDirectionLength = length(normalsSum);
    float3 coneDirection = normalsSum / coneDirectionLength;
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN)
    {
        cosConeAngle = 1.0f;
        for (uint i = 0; i < node.triangleCount; i++)
        {
            uint triangleIndex = gLightBVH.getNodeTriangleIndex(node, i);
            float3 normal = gLights.getTriangle(triangleIndex).normal;
            float cosDiffAngle = dot(coneDirection, normal);
            cosConeAngle = min(cosConeAngle, cosDiffAngle);
        }
        cosConeAngle = max(cosConeAngle, -1.f); // Guard against numerical errors
    }

    node.attribs.cosConeAngle = cosConeAngle;
    node.attribs.coneDirection = coneDirection;

    // Store the updated node.
    gLightBVH.setBLASLeafNode(nodeIndex, node);
}



/** Compute shader for refitting the BLAS internal nodes.
    This should be executed after updateLeafNodes().
*/

[numthreads(256, 1, 1)]
void updateBLASInternalNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;
    uint count = DTid.x;
    uint i = 0;
    for(i=0; i<gLights.meshCount ; i++){
        if(gNodeCountVec[i] == 0)continue;
        if(count > gNodeCountVec[i] )
            count -= gNodeCountVec[i];
        else
            break;
    }
    uint nodeIndex = gNodeIndices[gFirstNodeOffsetVec[i] + count];
    InternalNode node = gLightBVH.getBLASInternalNode(nodeIndex);

    uint leftChildIndex = nodeIndex + 1; // Left child is stored immediately after.
    uint rightChildIndex = node.rightChildIdx;

    const SharedNodeAttributes leftNode = gLightBVH.BLAS[leftChildIndex].getNodeAttributes();
    const SharedNodeAttributes rightNode = gLightBVH.BLAS[rightChildIndex].getNodeAttributes();

    // Update the node bounding box.
    float3 leftAabbMin, leftAabbMax;
    float3 rightAabbMin, rightAabbMax;
    leftNode.getAABB(leftAabbMin, leftAabbMax);
    rightNode.getAABB(rightAabbMin, rightAabbMax);

    float3 aabbMin = min(leftAabbMin, rightAabbMin);
    float3 aabbMax = max(leftAabbMax, rightAabbMax);

    node.attribs.setAABB(aabbMin, aabbMax);

    // Update the normal bounding cone.
    float3 coneDirectionSum = leftNode.coneDirection + rightNode.coneDirection;
    float coneDirectionLength = length(coneDirectionSum);
    float3 coneDirection = coneDirectionSum / coneDirectionLength;
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN &&
        leftNode.cosConeAngle != kInvalidCosConeAngle && rightNode.cosConeAngle != kInvalidCosConeAngle)
    {
        // This code rotates (cosLeftDiffAngle, sinLeftDiffAngle) counterclockwise by the left child's
        // cone spread angle, and similarly for the right child's cone.
        float cosLeftDiffAngle = dot(coneDirection, leftNode.coneDirection);
        float sinLeftDiffAngle = sinFromCos(cosLeftDiffAngle);

        float cosRightDiffAngle = dot(coneDirection, rightNode.coneDirection);
        float sinRightDiffAngle = sinFromCos(cosRightDiffAngle);

        float sinLeftConeAngle = sinFromCos(leftNode.cosConeAngle);
        float sinRightConeAngle = sinFromCos(rightNode.cosConeAngle);

        float sinLeftTotalAngle = sinLeftConeAngle * cosLeftDiffAngle + sinLeftDiffAngle * leftNode.cosConeAngle;
        float sinRightTotalAngle = sinRightConeAngle * cosRightDiffAngle + sinRightDiffAngle * rightNode.cosConeAngle;

        // If neither sum of angles is greater than pi, compute the new cosConeAngle.
        // Otherwise, deactivate the orientation cone as useless since it would represent the whole sphere.
        if (sinLeftTotalAngle > 0.0f && sinRightTotalAngle > 0.0f)
        {
            const float cosLeftTotalAngle = leftNode.cosConeAngle * cosLeftDiffAngle - sinLeftConeAngle * sinLeftDiffAngle;
            const float cosRightTotalAngle = rightNode.cosConeAngle * cosRightDiffAngle - sinRightConeAngle * sinRightDiffAngle;

            cosConeAngle = min(cosLeftTotalAngle, cosRightTotalAngle);
            cosConeAngle = max(cosConeAngle, -1.f); // Guard against numerical errors
        }
    }

    node.attribs.cosConeAngle = cosConeAngle;
    node.attribs.coneDirection = coneDirection;

    // Store the updated node.
    gLightBVH.setBLASInternalNode(nodeIndex, node);

}




/** Compute shader for refitting the TLAS leaf nodes.
    The code assumes a leaf stores an indexed list of emissive triangles.
*/

[numthreads(256, 1, 1)]
void updateTLASLeafNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;
    uint i = 0;
    uint nodeIndex = gNodeIndices[gFirstNodeOffset + DTid.x];
    LeafNode node = gLightBVH.getTLASLeafNode(nodeIndex);

    InternalNode childNode = gLightBVH.getBLASInternalNode(node.triangleOffset);

    //if (childNode.attribs.cosConeAngle != node.attribs.cosConeAngle ||
    //childNode.attribs.coneDirection.x  != node.attribs.coneDirection.x  || childNode.attribs.coneDirection.y != node.attribs.coneDirection.y|| childNode.attribs.coneDirection.z != node.attribs.coneDirection.z ||
    //childNode.attribs.origin.x != node.attribs.origin.x || childNode.attribs.origin.y != node.attribs.origin.y || childNode.attribs.origin.z != node.attribs.origin.z ||
    //childNode.attribs.extent.x != node.attribs.extent.x || childNode.attribs.extent.y != node.attribs.extent.y || childNode.attribs.extent.z != node.attribs.extent.z )
    //{
        // Update the node bounding box.
        float3 aabbMin = float3(FLT_MAX);
        float3 aabbMax = float3(-FLT_MAX);

        childNode.attribs.getAABB(aabbMin, aabbMax);
        node.attribs.setAABB(aabbMin, aabbMax);

        // Update the normal bounding cone.

        node.attribs.cosConeAngle = childNode.attribs.cosConeAngle;
        node.attribs.coneDirection = childNode.attribs.coneDirection;

        node.attribs.flux = childNode.attribs.flux;

        // Store the updated node.
        LeafNode origNode = gLightBVH.getTLASLeafNode(nodeIndex);
         
        for(i = 0; i < gLights.meshCount;i++)
        {
            if(gLightBVH.lightNodeIndices[i] == node.triangleOffset)
            {
                gLightBVH.setChangedLightNode(i,node);
                break;
            }
        }
        
        gLightBVH.setTLASLeafNode(nodeIndex, node);
    //}
}



/** Compute shader for refitting the TLAS internal nodes.
    This should be executed after updateLeafNodes().
*/

[numthreads(256, 1, 1)]
void updateTLASInternalNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;

    uint nodeIndex = gNodeIndices[gFirstNodeOffset + DTid.x];
    InternalNode node = gLightBVH.getTLASInternalNode(nodeIndex);

    uint leftChildIndex = nodeIndex + 1; // Left child is stored immediately after.
    uint rightChildIndex = node.rightChildIdx;

    const SharedNodeAttributes leftNode = gLightBVH.TLAS[leftChildIndex].getNodeAttributes();
    const SharedNodeAttributes rightNode = gLightBVH.TLAS[rightChildIndex].getNodeAttributes();

    // Update the node bounding box.
    float3 leftAabbMin, leftAabbMax;
    float3 rightAabbMin, rightAabbMax;
    leftNode.getAABB(leftAabbMin, leftAabbMax);
    rightNode.getAABB(rightAabbMin, rightAabbMax);

    float3 aabbMin = min(leftAabbMin, rightAabbMin);
    float3 aabbMax = max(leftAabbMax, rightAabbMax);

    node.attribs.setAABB(aabbMin, aabbMax);

    // Update the normal bounding cone.
    float3 coneDirectionSum = leftNode.coneDirection + rightNode.coneDirection;
    float coneDirectionLength = length(coneDirectionSum);
    float3 coneDirection = coneDirectionSum / coneDirectionLength;
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN &&
        leftNode.cosConeAngle != kInvalidCosConeAngle && rightNode.cosConeAngle != kInvalidCosConeAngle)
    {
        // This code rotates (cosLeftDiffAngle, sinLeftDiffAngle) counterclockwise by the left child's
        // cone spread angle, and similarly for the right child's cone.
        float cosLeftDiffAngle = dot(coneDirection, leftNode.coneDirection);
        float sinLeftDiffAngle = sinFromCos(cosLeftDiffAngle);

        float cosRightDiffAngle = dot(coneDirection, rightNode.coneDirection);
        float sinRightDiffAngle = sinFromCos(cosRightDiffAngle);

        float sinLeftConeAngle = sinFromCos(leftNode.cosConeAngle);
        float sinRightConeAngle = sinFromCos(rightNode.cosConeAngle);

        float sinLeftTotalAngle = sinLeftConeAngle * cosLeftDiffAngle + sinLeftDiffAngle * leftNode.cosConeAngle;
        float sinRightTotalAngle = sinRightConeAngle * cosRightDiffAngle + sinRightDiffAngle * rightNode.cosConeAngle;

        // If neither sum of angles is greater than pi, compute the new cosConeAngle.
        // Otherwise, deactivate the orientation cone as useless since it would represent the whole sphere.
        if (sinLeftTotalAngle > 0.0f && sinRightTotalAngle > 0.0f)
        {
            const float cosLeftTotalAngle = leftNode.cosConeAngle * cosLeftDiffAngle - sinLeftConeAngle * sinLeftDiffAngle;
            const float cosRightTotalAngle = rightNode.cosConeAngle * cosRightDiffAngle - sinRightConeAngle * sinRightDiffAngle;

            cosConeAngle = min(cosLeftTotalAngle, cosRightTotalAngle);
            cosConeAngle = max(cosConeAngle, -1.f); // Guard against numerical errors
        }
    }

    node.attribs.cosConeAngle = cosConeAngle;
    node.attribs.coneDirection = coneDirection;

    // Store the updated node.
    gLightBVH.setTLASInternalNode(nodeIndex, node);
}



// Returns sin(a) based on cos(a) for a in [0,pi].
float sinFromCos(float cosAngle)
{
    return sqrt(max(0.0f, 1.0f - cosAngle * cosAngle));
}



