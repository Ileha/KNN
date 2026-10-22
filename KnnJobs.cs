using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

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

    public struct IntervalData
    {
        public float3 Min;
        public float3 Max;
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
    public struct KnnBatchIntervalJob : IJobParallelForBatch
    {
        [ReadOnly] public KnnContainer Container;
        [ReadOnly] public NativeArray<IntervalData> Input;
        public NativeStream.Writer Output;

        public void Execute(int startIndex, int count)
        {
            // One buffer per batch; it keeps its grown capacity across requests
            var hits = new NativeList<int>(Allocator.Temp);
            for (var index = startIndex; index < startIndex + count; index++)
            {
                var request = Input[index];
                hits.Clear();
                Container.QueryInterval(request.Min, request.Max, hits);

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
            // One buffer and one query temp per batch, resized per request
            var result = new NativeList<int>(Allocator.Temp);
            var temp = KnnContainer.KnnQueryTemp.Create(1);
            for (var index = startIndex; index < startIndex + count; index++)
            {
                var request = Input[index];
                // Clamped, so the result has no -1 slots for missing points
                var k = math.min(request.K, Container.Points.Length);

                Output.BeginForEachIndex(index);
                if (k > 0)
                {
                    result.ResizeUninitialized(k);
                    Container.QueryKNearest(request.Position, result.AsArray(), ref temp);

                    // KNN returns the farthest first
                    for (var i = k - 1; i >= 0; i--)
                    {
                        Output.Write(result[i]);
                    }
                }

                Output.EndForEachIndex();
            }

            temp.Dispose();
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