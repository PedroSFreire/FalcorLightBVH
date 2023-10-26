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
#include "TwoLevelLightBVHSampler.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtx/io.hpp>
#include <algorithm>
#include <numeric>




namespace Falcor
{

    void asyncRebuild(TwoLevelLightBVH::SharedPtr mpBVH, TwoLevelLightBVHBuilder::SharedPtr mpBVHBuilder) {
        mpBVHBuilder->reBuild(*mpBVH);
    }

    namespace
    {
        const Gui::DropdownList kSolidAngleBoundList =
        {
            { (uint32_t)SolidAngleBoundMethod::Sphere, "Sphere" },
            { (uint32_t)SolidAngleBoundMethod::BoxToCenter, "Cone around center dir" },
            { (uint32_t)SolidAngleBoundMethod::BoxToAverage, "Cone around average dir" },
        };
    }

    TwoLevelLightBVHSampler::SharedPtr TwoLevelLightBVHSampler::create(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options)
    {
        return SharedPtr(new TwoLevelLightBVHSampler(pRenderContext, pScene, options));
    }

    void TwoLevelLightBVHSampler::unlockRebuildMutex() {
        mpBVH->uploadGPUMutex.unlock();
    }


    bool TwoLevelLightBVHSampler::update(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE("TwoLevelLightBVHSampler::update");

        bool samplerChanged = false;
        bool needsRefit = false;
        //if (mpBVH->threadOn)
          //  mpBVH->uploadGPUMutex.unlock();
        // Check if light collection has changed.
        if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::LightCollectionChanged))
        {
            if (mOptions.buildOptions.allowRefitting && !mNeedsRebuild) needsRefit = true;
            else
                mNeedsRebuild = true;
        }

        if (mpBVH->threadOn) {
            mpBVH->rebuildThread.join();
            mpBVH->threadOn = false;
        }
        if (samplerChanged == false && needsRefit == false && getBVH() != NULL ) {
            printf("\n");
            printf("Rebuild Time : %f  \n",getBVH().get()->TotalRebuildTime / 300);
            printf("Refit Time : %f  \n", getBVH().get()->TotalRefitTime / 300);
            printf("\n");
            writen = true;
        }
        // Rebuild BVH if it's marked as dirty.
        if (mNeedsRebuild)
        {
            mpBVHBuilder->build(*mpBVH);
            mNeedsRebuild = false;
            samplerChanged = true;
        }
        else if (needsRefit)
        {

            mpBVH->updateRefitData(pRenderContext);
   
            if (mpBVH->refit(pRenderContext)) {
                //use commented code for synced rebuild
                //mpBVHBuilder->reBuild(*mpBVH);
                mpBVH->uploadGPUMutex.lock();
                mpBVH->rebuildThread = std::thread{ asyncRebuild , mpBVH , mpBVHBuilder };
                mpBVH->threadOn = true;

                samplerChanged = true;
            }

           
        }
        else {
            if (mpBVH->threadOn) {
                mpBVH->threadOn = false;
                mpBVH->uploadGPUMutex.unlock();
                mpBVH->rebuildThread.join();
            }
            
        }

        return samplerChanged;
    }

    Program::DefineList TwoLevelLightBVHSampler::getDefines() const
    {
        // Call the base class first.
        auto defines = EmissiveLightSampler::getDefines();

        // Add our defines. None of these change the program vars.
        defines.add("_USE_BOUNDING_CONE", mOptions.useBoundingCone ? "1" : "0");
        defines.add("_USE_LIGHTING_CONE", mOptions.useLightingCone ? "1" : "0");
        defines.add("_DISABLE_NODE_FLUX", mOptions.disableNodeFlux ? "1" : "0");
        defines.add("_USE_UNIFORM_TRIANGLE_SAMPLING", mOptions.useUniformTriangleSampling ? "1" : "0");
        defines.add("_ACTUAL_MAX_TRIANGLES_PER_NODE", std::to_string(mOptions.buildOptions.maxTriangleCountPerLeaf));
        defines.add("_SOLID_ANGLE_BOUND_METHOD", std::to_string((uint32_t)mOptions.solidAngleBoundMethod));

        return defines;
    }

    void TwoLevelLightBVHSampler::setShaderData(const ShaderVar& var) const
    {
        FALCOR_ASSERT(var.isValid());
        FALCOR_ASSERT(mpBVH);
        mpBVH->setShaderData(var["_twoLevelLightBVH"]);
    }

    bool TwoLevelLightBVHSampler::renderUI(Gui::Widgets& widgets)
    {
        bool optionsChanged = false;
        if (auto buildGroup = widgets.group("BVH building options"))
        {
            if (mpBVHBuilder->renderUI(buildGroup))
            {
                mOptions.buildOptions = mpBVHBuilder->getOptions();
                mNeedsRebuild = optionsChanged = true;
            }
        }

        if (auto traversalGroup = widgets.group("BVH traversal options"))
        {
            optionsChanged |= traversalGroup.checkbox("Use bounding cone (NdotL)", mOptions.useBoundingCone);
            if (traversalGroup.checkbox("Use lighting cone", mOptions.useLightingCone))
            {
                mNeedsRebuild = optionsChanged = true;
            }
            optionsChanged |= traversalGroup.checkbox("Disable node flux", mOptions.disableNodeFlux);
            optionsChanged |= traversalGroup.checkbox("Use triangle uniform sampling", mOptions.useUniformTriangleSampling);

            if (traversalGroup.dropdown("Solid Angle Bound", kSolidAngleBoundList, (uint32_t&)mOptions.solidAngleBoundMethod))
            {
                mNeedsRebuild = optionsChanged = true;
            }
            traversalGroup.tooltip("Selects the bounding method for the dot(N,L) term:\n\n"
                "Sphere - Use a bounding sphere around the AABB. This is the fastest, but least conservative method.\n"
                "Cone around center dir - Compute a bounding cone around the direction to the center of the AABB. This is more expensive, but gives tighter bounds.\n"
                "Cone around average dir - Computes a bounding cone to the average direction of all AABB corners. This is the most expensive, but gives the tightest bounds.");
        }


        if (auto statGroup = widgets.group("BVH statistics"))
        {
            mpBVH->renderUI(statGroup);
        }

        return optionsChanged;
    }

    TwoLevelLightBVH::SharedConstPtr TwoLevelLightBVHSampler::getBVH() const
    {
        return mpBVH->isValid() ? mpBVH : nullptr;
    }

    TwoLevelLightBVHSampler::TwoLevelLightBVHSampler(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options)
        : EmissiveLightSampler(EmissiveLightSamplerType::TwoLevelLightBVH, pScene)
        , mOptions(options)
    {
        // Create the BVH and builder.
        mpBVHBuilder = TwoLevelLightBVHBuilder::create(mOptions.buildOptions);
        if (!mpBVHBuilder) throw RuntimeError("Failed to create BVH builder");

        mpBVH = TwoLevelLightBVH::create(pScene->getLightCollection(pRenderContext));
        if (!mpBVH) throw RuntimeError("Failed to create BVH");
    }

    FALCOR_SCRIPT_BINDING(TwoLevelLightBVHSampler)
    {
        //pybind11::enum_<SolidAngleBoundMethod> solidAngleBoundMethod(m, "SolidAngleBoundMethod");
        //solidAngleBoundMethod.value("BoxToAverage", SolidAngleBoundMethod::BoxToAverage);
        //solidAngleBoundMethod.value("BoxToCenter", SolidAngleBoundMethod::BoxToCenter);
        //solidAngleBoundMethod.value("Sphere", SolidAngleBoundMethod::Sphere);

        // TODO use a nested class in the bindings when supported.
        ScriptBindings::SerializableStruct<TwoLevelLightBVHSampler::Options> options(m, "TwoLevelLightBVHSamplerOptions");
#define field(f_) field(#f_, &TwoLevelLightBVHSampler::Options::f_)
        options.field(buildOptions);
        options.field(useBoundingCone);
        options.field(useLightingCone);
        options.field(disableNodeFlux);
        options.field(useUniformTriangleSampling);
        options.field(solidAngleBoundMethod);
#undef field
    }
}
