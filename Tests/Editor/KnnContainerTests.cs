using System.Collections.Generic;
using System.Linq;
using KNN.Jobs;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace KNN.Tests
{
    // Ported from Supercluster.KDTree (AccuracyTest), codeandcats/KdTree (KdTreeTests)
    // and viliwonka/KDTree (KDTreeQueryTests, KDTreeQueryResizeTests)
    public class KnnContainerTests
    {
        private const uint Seed = 12345;
        private const int DataSize = 10000;
        private const int TestDataSize = 100;
        private const float Range = 1000f;

        // Distances are from Toowoomba in (longitude, -latitude) space
        private static readonly (string Name, float3 Position, float Distance)[] Cities =
        {
            ("Toowoomba", new float3(151.8643252f, 27.5829487f, 0f), 0f),
            ("Brisbane", new float3(153.0234489f, 27.4710107f, 0f), 1.16451615177537f),
            ("Goldcoast", new float3(153.4256987f, 28.0172605f, 0f), 1.6206523211724f),
            ("Sunshine", new float3(153.0554193f, 27.3748288f, 0f), 1.20913979664506f),
            ("Melbourne", new float3(144.96328f, 37.814107f, 0f), 12.3410301438779f),
            ("Sydney", new float3(151.2069902f, 33.8674869f, 0f), 6.31882185929341f),
            ("Perth", new float3(115.8574693f, 31.9530044f, 0f), 36.2710774395312f),
            ("Darwin", new float3(130.8417694f, 12.4628198f, 0f), 25.895292049265f),
        };

        private NativeArray<float3> _points;
        private KnnContainer _knn;
        private bool _isBuilt;

        [TearDown]
        public void TearDown()
        {
            if (_isBuilt)
            {
                _knn.Dispose();
                _isBuilt = false;
            }

            if (_points.IsCreated)
            {
                _points.Dispose();
            }
        }

        [Test]
        public void QueryKNearest_Cities_ReturnsNearestInOrder()
        {
            Build(Cities.Select(c => c.Position).ToArray());
            var expected = Cities.OrderBy(c => c.Distance).Select(c => c.Name).ToArray();

            for (var k = 1; k <= Cities.Length; k++)
            {
                var actual = QueryKNearest(Cities[0].Position, k).Select(i => Cities[i].Name).ToArray();

                CollectionAssert.AreEqual(expected.Take(k).ToArray(), actual);
            }
        }

        [Test]
        public void QueryRange_Cities_ReturnsCitiesInRadiusInOrder()
        {
            Build(Cities.Select(c => c.Position).ToArray());

            for (var radius = 1; radius < 100; radius *= 2)
            {
                var expected = Cities.Where(c => c.Distance <= radius).OrderBy(c => c.Distance).Select(c => c.Name).ToArray();
                var actual = QueryRange(Cities[0].Position, radius).Select(i => Cities[i].Name).ToArray();

                CollectionAssert.AreEqual(expected, actual, $"radius {radius}");
            }
        }

        [Test]
        public void QueryKNearest_RandomPoints_MatchesLinearSearch()
        {
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));

            for (var i = 0; i < TestDataSize; i++)
            {
                AssertKNearestMatchesLinear(random.NextFloat3() * Range, 1);
                AssertKNearestMatchesLinear(random.NextFloat3() * Range, 10);
            }
        }

        [Test]
        public void QueryRange_RandomPoints_MatchesLinearSearch()
        {
            const float radius = 100f;
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));

            for (var i = 0; i < TestDataSize; i++)
            {
                AssertRangeMatchesLinear(random.NextFloat3() * Range, radius);
            }
        }

        [Test]
        public void QueryInterval_RandomPoints_MatchesLinearSearch()
        {
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));
            var result = new NativeList<int>(Allocator.TempJob);

            for (var i = 0; i < TestDataSize; i++)
            {
                var min = random.NextFloat3() * Range;
                var max = min + random.NextFloat3() * Range * 0.2f;
                result.Clear();
                new QueryIntervalJob { Container = _knn, Min = min, Max = max, Result = result }.Schedule().Complete();

                var expected = Enumerable.Range(0, _points.Length)
                    .Where(p => math.all(_points[p] >= min & _points[p] <= max))
                    .ToArray();
                CollectionAssert.AreEquivalent(expected, result.AsArray().ToArray());
            }

            result.Dispose();
        }

        [Test]
        public void Rebuild_AfterPointsMove_MatchesLinearSearch()
        {
            const int frames = 5;
            const float step = 50f;
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));

            for (var frame = 0; frame < frames; frame++)
            {
                for (var i = 0; i < _points.Length; i++)
                {
                    _points[i] += (random.NextFloat3() - 0.5f) * step;
                }

                _knn.Rebuild();

                for (var i = 0; i < TestDataSize; i++)
                {
                    var query = random.NextFloat3() * Range;
                    AssertKNearestMatchesLinear(query, 10);
                    AssertRangeMatchesLinear(query, 100f);
                }
            }
        }

        [Test]
        public void Rebuild_AfterPointsGrow_MatchesLinearSearch()
        {
            const int frames = 100;
            const int pointsPerFrame = 3;
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, 1));

            for (var frame = 0; frame < frames; frame++)
            {
                var grown = new NativeArray<float3>(_points.Length + pointsPerFrame, Allocator.Persistent);
                NativeArray<float3>.Copy(_points, grown, _points.Length);
                for (var i = _points.Length; i < grown.Length; i++)
                {
                    grown[i] = random.NextFloat3() * Range;
                }

                _points.Dispose();
                _points = grown;
                _knn.Points = grown;
                _knn.Rebuild();

                var query = random.NextFloat3() * Range;
                AssertKNearestMatchesLinear(query, 1);
                AssertKNearestMatchesLinear(query, _points.Length);
                AssertRangeMatchesLinear(query, 300f);
            }
        }

        private void Build(float3[] points)
        {
            _points = new NativeArray<float3>(points, Allocator.Persistent);
            _knn = new KnnContainer(_points, true, Allocator.Persistent);
            _isBuilt = true;
        }

        private static float3[] GeneratePoints(ref Random random, int count)
        {
            var points = new float3[count];
            for (var i = 0; i < count; i++)
            {
                points[i] = random.NextFloat3() * Range;
            }

            return points;
        }

        // Nearest first
        private int[] QueryKNearest(float3 query, int k)
        {
            var result = new NativeArray<int>(k, Allocator.Temp);
            _knn.QueryKNearest(query, result);
            var indices = result.Reverse().ToArray();
            result.Dispose();
            return indices;
        }

        // Nearest first
        private int[] QueryRange(float3 query, float radius)
        {
            var result = new NativeList<int>(Allocator.Temp);
            _knn.QueryRange(query, radius, result);
            var indices = result.AsArray().Reverse().ToArray();
            result.Dispose();
            return indices;
        }

        // Compares distances, not indices, so equally distant points can come in any order
        private void AssertKNearestMatchesLinear(float3 query, int k)
        {
            var expected = LinearDistances(query).OrderBy(d => d).Take(k).ToArray();
            var actual = QueryKNearest(query, k).Select(i => math.lengthsq(_points[i] - query)).ToArray();

            CollectionAssert.AreEqual(expected, actual, $"query {query}, k {k}");
        }

        private void AssertRangeMatchesLinear(float3 query, float radius)
        {
            var expected = Enumerable.Range(0, _points.Length)
                .Where(i => math.lengthsq(_points[i] - query) <= radius * radius)
                .ToArray();
            var actual = QueryRange(query, radius);

            CollectionAssert.AreEquivalent(expected, actual, $"query {query}, radius {radius}");
        }

        private IEnumerable<float> LinearDistances(float3 query)
        {
            for (var i = 0; i < _points.Length; i++)
            {
                yield return math.lengthsq(_points[i] - query);
            }
        }
    }
}
