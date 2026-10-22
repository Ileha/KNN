using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnsafeUtilityEx = KNN.Internal.UnsafeUtilityEx;

namespace KNN.Jobs
{
    public struct RadiusData
    {
        public float3 Position;
        public float Radius;
    }

    public struct KNearestData
    {
        public float3 Position;
        public int K;
    }

    // Top-level and non-generic so Burst can compile them ahead of time in player builds
    [BurstCompile]
    public struct KnnBatchRadiusJob : IJobParallelForBatch
    {
        [ReadOnly] public KnnContainer Container;
        [ReadOnly] public NativeArray<RadiusData> Input;
        public NativeStream.Writer Output;

        public void Execute(int startIndex, int count)
        {
            // One buffer per batch; it keeps its grown capacity across requests
            var hits = new NativeList<int>(Allocator.Temp);
            for (var index = startIndex; index < startIndex + count; index++)
            {
                var request = Input[index];
                hits.Clear();
                Container.QueryRange(request.Position, request.Radius, hits);

                Output.BeginForEachIndex(index);
                for (var i = 0; i < hits.Length; i++)
                {
                    Output.Write(hits[i]);
                }

                Output.EndForEachIndex();
            }

            hits.Dispose();
        }
    }

    [BurstCompile]
    public struct KnnBatchKNearestJob : IJobParallelForBatch
    {
        [ReadOnly] public KnnContainer Container;
        [ReadOnly] public NativeArray<KNearestData> Input;
        public NativeStream.Writer Output;

        public void Execute(int startIndex, int count)
        {
            // One buffer per batch, resized per request
            var result = new NativeList<int>(Allocator.Temp);
            for (var index = startIndex; index < startIndex + count; index++)
            {
                var request = Input[index];
                // KNN reads past its heap when k exceeds the point count
                var k = math.min(request.K, Container.Points.Length);

                Output.BeginForEachIndex(index);
                if (k > 0)
                {
                    result.ResizeUninitialized(k);
                    Container.QueryKNearest(request.Position, result.AsArray());

                    // KNN returns the farthest first
                    for (var i = k - 1; i >= 0; i--)
                    {
                        Output.Write(result[i]);
                    }
                }

                Output.EndForEachIndex();
            }

            result.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct QueryKNearestJob : IJob
    {
        [ReadOnly] private KnnContainer m_container;
        [WriteOnly] private readonly NativeSlice<int> m_result;

        private readonly float3 m_queryPosition;

        public QueryKNearestJob(KnnContainer container, float3 queryPosition, NativeSlice<int> result)
        {
            m_result = result;
            m_queryPosition = queryPosition;
            m_container = container;
        }

        void IJob.Execute()
        {
            m_container.QueryKNearest(m_queryPosition, m_result);
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct QueryRangeJob : IJob
    {
        [ReadOnly] private KnnContainer m_container;
        [WriteOnly] private readonly NativeList<int> m_result;

        private readonly float m_range;
        private readonly float3 m_queryPosition;

        public QueryRangeJob(KnnContainer container, float3 queryPosition, float range, NativeList<int> result)
        {
            m_result = result;
            m_range = range;
            m_queryPosition = queryPosition;
            m_container = container;
        }

        void IJob.Execute()
        {
            m_container.QueryRange(m_queryPosition, m_range, m_result);
        }
    }

    [BurstCompile]
    public struct QueryIntervalJob : IJob
    {
        [ReadOnly] public KnnContainer Container;
        public float3 Min;
        public float3 Max;
        public NativeList<int> Result;

        void IJob.Execute()
        {
            Container.QueryInterval(Min, Max, Result);
        }
    }


    [BurstCompile(CompileSynchronously = true)]
    public struct QueryKNearestBatchJob : IJobParallelForBatch
    {
        [ReadOnly] private KnnContainer m_container;
        [ReadOnly] private NativeSlice<float3> m_queryPositions;

        // Unity really doesn't like it when we write to the same underlying array
        // Even if slices don't overlap... So we're just being dangerous here
        [NativeDisableParallelForRestriction] [NativeDisableContainerSafetyRestriction]
        private readonly NativeSlice<int> m_results;

        private readonly int m_k;

        public QueryKNearestBatchJob(KnnContainer container, NativeArray<float3> queryPositions,
            NativeSlice<int> results)
        {
            m_container = container;
            m_queryPositions = queryPositions;
            m_results = results;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            if (queryPositions.Length == 0 || results.Length % queryPositions.Length != 0)
            {
                Debug.LogError("Make sure your results array is a multiple in length of your querypositions array!");
            }
#endif

            m_k = results.Length / queryPositions.Length;
        }

        public void Execute(int startIndex, int count)
        {
            // Write results to proper slice!
            for (var index = startIndex; index < startIndex + count; ++index)
            {
                var resultsSlice = m_results.Slice(index * m_k, m_k);
                m_container.QueryKNearest(m_queryPositions[index], resultsSlice);
            }
        }
    }

    public unsafe struct RangeQueryResult
    {
        public int Length;

        private readonly int* m_indices;
        private readonly int m_capacity;

        private readonly Allocator m_allocator;

        public int this[int index]
        {
            get
            {
                if (index >= m_capacity)
                {
                    throw new IndexOutOfRangeException();
                }

                return UnsafeUtility.ReadArrayElement<int>(m_indices, index);
            }
        }

        public RangeQueryResult(int maxCount, Allocator allocator)
        {
            m_capacity = maxCount;
            m_indices = UnsafeUtilityEx.AllocArray<int>(m_capacity, allocator);
            Length = 0;
            m_allocator = allocator;
        }

        public void SetResults(NativeList<int> result)
        {
            UnsafeUtility.MemCpy(m_indices, result.GetUnsafePtr(), Mathf.Min(m_capacity, result.Length) * sizeof(int));
            Length = Mathf.Min(m_capacity, result.Length);
        }

        public void Dispose()
        {
            UnsafeUtility.Free(m_indices, m_allocator);
        }
    }


    [BurstCompile(CompileSynchronously = true)]
    public struct QueryRangeBatchJob : IJobParallelForBatch
    {
        [ReadOnly] private KnnContainer m_container;
        [ReadOnly] private NativeSlice<float3> m_queryPositions;

        private readonly float m_range;

        public NativeArray<RangeQueryResult> Results;

        public QueryRangeBatchJob(KnnContainer container, NativeArray<float3> queryPositions, float range,
            NativeArray<RangeQueryResult> results)
        {
            m_container = container;
            m_queryPositions = queryPositions;
            m_range = range;
            Results = results;
        }

        public void Execute(int startIndex, int count)
        {
            // Write results to proper slice!
            for (var index = startIndex; index < startIndex + count; ++index)
            {
                var tempList = new NativeList<int>(Allocator.Temp);
                m_container.QueryRange(m_queryPositions[index], m_range, tempList);

                var result = Results[index];
                result.SetResults(tempList);

                Results[index] = result;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct KnnRebuildJob : IJob
    {
        private KnnContainer m_container;

        public KnnRebuildJob(KnnContainer container)
        {
            m_container = container;
        }

        void IJob.Execute()
        {
            m_container.Rebuild();
        }
    }
}