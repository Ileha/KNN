using System.Collections.Generic;
using System.Linq;
using KNN.Jobs;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace KNN.Tests
{
    // Ported from Supercluster.KDTree (AccuracyTest), codeandcats/KdTree (KdTreeTests),
    // viliwonka/KDTree (KDTreeQueryTests, KDTreeQueryResizeTests), scipy (test_kdtree.py)
    // and NearestNeighbors.jl (test_knn.jl, test_inrange.jl)
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

        // codeandcats/KdTree TestGetNearestNeighbours
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

        // codeandcats/KdTree TestRadialSearch
        [Test]
        public void QueryRange_Cities_ReturnsCitiesInRadius()
        {
            Build(Cities.Select(c => c.Position).ToArray());

            for (var radius = 1; radius < 100; radius *= 2)
            {
                var expected = Cities.Where(c => c.Distance <= radius).Select(c => c.Name).ToArray();
                var actual = QueryRange(Cities[0].Position, radius).Select(i => Cities[i].Name).ToArray();

                CollectionAssert.AreEquivalent(expected, actual, $"radius {radius}");
            }
        }

        // Supercluster.KDTree FindNearestNeighborTest
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

        // Supercluster.KDTree RadialSearchTest
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

        // viliwonka/KDTree KDTreeQueryTests (Interval)
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
                new QueryIntervalJob { Container = _knn, Min = min, Max = max, Result = result }.Run();

                var expected = Enumerable.Range(0, _points.Length)
                    .Where(p => math.all(_points[p] >= min & _points[p] <= max))
                    .ToArray();
                CollectionAssert.AreEquivalent(expected, result.AsArray().ToArray());
            }

            result.Dispose();
        }

        // no source: the old interval query searched the sphere around the box, which can miss corners on rounding
        [Test]
        public void QueryInterval_IntegerGrid_IncludesPointsOnFacesAndCorners()
        {
            const int size = 6;
            var points = new List<float3>();
            for (var x = 0; x < size; x++)
            {
                for (var y = 0; y < size; y++)
                {
                    for (var z = 0; z < size; z++)
                    {
                        points.Add(new float3(x, y, z));
                    }
                }
            }

            Build(points.ToArray());
            var min = new float3(1f);
            var max = new float3(3f);
            var result = new NativeList<int>(Allocator.TempJob);
            new QueryIntervalJob { Container = _knn, Min = min, Max = max, Result = result }.Run();

            var expected = Enumerable.Range(0, points.Count).Where(i => math.all(points[i] >= min & points[i] <= max)).ToArray();
            CollectionAssert.AreEquivalent(expected, result.AsArray().ToArray());
            result.Dispose();
        }

        // viliwonka/KDTree KDTreeQueryTests (Update + Rebuild)
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

        // viliwonka/KDTree KDTreeQueryResizeTests
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

        // no source: a wide query has hundreds of nodes to visit, more than the old fixed min heap of 64
        [Test]
        public void QueryRange_LargeRadiusOnLargeCloud_MatchesLinearSearch()
        {
            const int count = 200000;
            const int queries = 3;
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, count));

            for (var i = 0; i < queries; i++)
            {
                AssertRangeMatchesLinear(random.NextFloat3() * Range, Range * 0.5f);
            }
        }

        // no source: points on the bounds edge used to make no progress when split on that axis
        [Test]
        public void Build_PointsSharingCoordinate_MatchesLinearSearch()
        {
            const int count = 1000;
            var random = new Random(Seed);
            var points = new float3[count * 2];
            for (var i = 0; i < count; i++)
            {
                points[i] = new float3(0f, random.NextFloat2());
                points[count + i] = new float3(10f, random.NextFloat2());
            }

            Build(points);

            for (var i = 0; i < TestDataSize; i++)
            {
                var query = random.NextFloat3(new float3(-1f), new float3(11f, 2f, 2f));
                AssertKNearestMatchesLinear(query, 10);
                AssertRangeMatchesLinear(query, 0.2f);
            }
        }

        // scipy test_kdtree_duplicated_inputs
        [Test]
        public void Build_DuplicatedPoints_QueriesMatchLinearSearch()
        {
            const int count = 1024;
            var points = new float3[count];
            for (var i = 0; i < count; i++)
            {
                points[i] = i < count / 2 ? new float3(1f) : new float3(2f);
            }

            Build(points);

            CollectionAssert.AreEquivalent(Enumerable.Range(0, count / 2), QueryRange(new float3(1f), 0.1f));
            CollectionAssert.AreEquivalent(Enumerable.Range(count / 2, count / 2), QueryRange(new float3(2f), 0.1f));
            AssertKNearestMatchesLinear(new float3(1f), 600);
            AssertKNearestMatchesLinear(new float3(1.4f), 600);
            AssertRangeMatchesLinear(new float3(1.5f), 1f);
        }

        // scipy test_onetree_query: a tiny cluster, a unit cluster and a shifted cluster
        [Test]
        public void Query_ClusteredPoints_MatchesLinearSearch()
        {
            const int clusterSize = 2000;
            const int queriesPerCluster = 30;
            float[] radii = { 0.1f, 0.001f, 0.00001f, 0.000001f };
            var random = new Random(Seed);
            var points = new float3[clusterSize * 3];
            for (var i = 0; i < clusterSize; i++)
            {
                points[i] = random.NextFloat3(-1f, 1f) * 0.001f;
                points[clusterSize + i] = random.NextFloat3(-1f, 1f) + 2f;
                points[clusterSize * 2 + i] = random.NextFloat3(-1f, 1f);
            }

            Build(points);

            for (var cluster = 0; cluster < 3; cluster++)
            {
                for (var i = 0; i < queriesPerCluster; i++)
                {
                    var query = _points[cluster * clusterSize + i];
                    AssertKNearestMatchesLinear(query, 10);
                    foreach (var radius in radii)
                    {
                        AssertRangeMatchesLinear(query, radius);
                    }
                }
            }
        }

        // scipy test_ball_point_ints: the radius bound is inclusive
        [Test]
        public void QueryRange_IntegerGrid_IncludesPointsOnRadius()
        {
            var points = new float3[16];
            for (var x = 0; x < 4; x++)
            {
                for (var y = 0; y < 4; y++)
                {
                    points[x * 4 + y] = new float3(x, y, 0f);
                }
            }

            Build(points);

            CollectionAssert.AreEquivalent(new[] { 4, 8, 9, 12 }, QueryRange(new float3(2f, 0f, 0f), 1f));
        }

        // NearestNeighbors.jl test_inrange: 8 node cube
        [Test]
        public void QueryRange_CubeCorners_ReturnsExpected()
        {
            Build(new[]
            {
                new float3(0, 0, 0), new float3(0, 0, 1), new float3(0, 1, 0), new float3(0, 1, 1),
                new float3(1, 0, 0), new float3(1, 0, 1), new float3(1, 1, 0), new float3(1, 1, 1),
            });

            CollectionAssert.AreEquivalent(new[] { 7 }, QueryRange(new float3(1.1f), 0.2f));
            CollectionAssert.AreEquivalent(new[] { 0, 1 }, QueryRange(new float3(0f, 0f, 0.5f), 0.6f));
            CollectionAssert.AreEquivalent(new[] { 0 }, QueryRange(new float3(0f), 0.6f));
            // The original uses 1/3, which puts corner 3 at distance 1 and fails on float rounding
            CollectionAssert.AreEquivalent(new[] { 0, 1, 2, 4 }, QueryRange(new float3(0.3f), 1f));
            CollectionAssert.IsEmpty(QueryRange(new float3(0.5f), 0.2f));
            CollectionAssert.AreEquivalent(Enumerable.Range(0, 8), QueryRange(new float3(0.5f), 1f));
        }

        // scipy test_len0_arrays
        [Test]
        public void QueryRange_RadiusBelowNearestDistance_ReturnsEmpty()
        {
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));

            for (var i = 0; i < TestDataSize; i++)
            {
                var query = random.NextFloat3() * Range;
                var nearest = _points[QueryKNearest(query, 1)[0]];

                CollectionAssert.IsEmpty(QueryRange(query, 0.1f * math.distance(nearest, query)));
            }
        }

        // scipy test_found_all_outside
        [Test]
        public void Query_OutsideCloud_MatchesLinearSearch()
        {
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));

            for (var i = 0; i < TestDataSize; i++)
            {
                var inside = random.NextFloat3() * Range;
                foreach (var query in new[] { inside + Range, inside - Range })
                {
                    AssertKNearestMatchesLinear(query, 10);
                    AssertRangeMatchesLinear(query, Range);
                }
            }
        }

        // scipy test_against_logic_error_regression: integer points give many ties and duplicates
        [Test]
        public void QueryRange_IntegerPoints_IsSymmetric()
        {
            const int count = 2000;
            const float cells = 10f;
            const float radius = 3f;
            var random = new Random(Seed);
            var points = new float3[count];
            for (var i = 0; i < count; i++)
            {
                points[i] = math.floor(random.NextFloat3() * cells);
            }

            Build(points);

            var neighbours = Enumerable.Range(0, count).Select(i => new HashSet<int>(QueryRange(_points[i], radius))).ToArray();
            var asymmetric = Enumerable.Range(0, count)
                .SelectMany(i => neighbours[i].Where(j => !neighbours[j].Contains(i)).Select(j => (i, j)))
                .ToArray();

            CollectionAssert.IsEmpty(asymmetric);
            AssertRangeMatchesLinear(_points[0], radius);
        }

        // bryding/KNN: k above the point count used to read past the heap
        [Test]
        public void QueryKNearest_KAbovePointCount_FillsMissingWithMinusOne()
        {
            Build(new[] { new float3(0f), new float3(1f), new float3(2f) });
            var result = new NativeArray<int>(5, Allocator.Temp);

            _knn.QueryKNearest(new float3(0f), result);

            // Farthest first, then the missing slots
            CollectionAssert.AreEqual(new[] { 2, 1, 0, -1, -1 }, result.ToArray());
            result.Dispose();
        }

        // no source: k = 0 used to read past the heap
        [Test]
        public void QueryKNearest_KZero_DoesNothing()
        {
            Build(new[] { new float3(0f), new float3(1f) });
            var result = new NativeArray<int>(0, Allocator.Temp);

            Assert.DoesNotThrow(() => _knn.QueryKNearest(new float3(0f), result));
            result.Dispose();
        }

        // bryding/KNN: a reused temp resizes to each k and matches a fresh query
        [Test]
        public void QueryKNearest_ReusedTemp_MatchesFreshQuery()
        {
            int[] ks = { 10, 3, 50, 3, 1 };
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize));
            var temp = KnnContainer.KnnQueryTemp.Create(1);

            foreach (var k in ks)
            {
                var query = random.NextFloat3() * Range;
                var reused = new NativeArray<int>(k, Allocator.Temp);
                var fresh = new NativeArray<int>(k, Allocator.Temp);

                _knn.QueryKNearest(query, reused, ref temp);
                _knn.QueryKNearest(query, fresh);

                CollectionAssert.AreEqual(fresh.ToArray(), reused.ToArray(), $"k {k}");
                reused.Dispose();
                fresh.Dispose();
            }

            temp.Dispose();
        }

        // NearestNeighbors.jl one_point_tree
        [Test]
        public void Query_OnePoint_AlwaysReturnsIt()
        {
            Build(new[] { new float3(0.5f) });

            for (var i = 0; i < 8; i++)
            {
                var corner = new float3(i & 1, (i >> 1) & 1, (i >> 2) & 1);

                CollectionAssert.AreEqual(new[] { 0 }, QueryKNearest(corner, 1));
                CollectionAssert.AreEqual(new[] { 0 }, QueryRange(corner, 1f));
            }
        }

        // scipy test_kdtree_build_modes: KNN has one build mode, so compare the deferred build instead
        [Test]
        public void KnnRebuildJob_DeferredBuild_MatchesLinearSearch()
        {
            var random = new Random(Seed);
            Build(GeneratePoints(ref random, DataSize), false);
            new KnnRebuildJob(_knn).Run();

            for (var i = 0; i < TestDataSize; i++)
            {
                var query = random.NextFloat3() * Range;
                AssertKNearestMatchesLinear(query, 10);
                AssertRangeMatchesLinear(query, 100f);
            }
        }

        private void Build(float3[] points, bool buildNow = true)
        {
            _points = new NativeArray<float3>(points, Allocator.Persistent);
            _knn = new KnnContainer(_points, buildNow, Allocator.Persistent);
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

        // Unsorted
        private int[] QueryRange(float3 query, float radius)
        {
            var result = new NativeList<int>(Allocator.Temp);
            _knn.QueryRange(query, radius, result);
            var indices = result.AsArray().ToArray();
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
            // Sorted AreEqual, because AreEquivalent is quadratic and stalls on large results
            var actual = QueryRange(query, radius).OrderBy(i => i).ToArray();

            CollectionAssert.AreEqual(expected, actual, $"query {query}, radius {radius}");
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
