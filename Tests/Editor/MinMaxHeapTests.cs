using System.Linq;
using KNN.Internal;
using NUnit.Framework;
using Unity.Collections;

namespace KNN.Tests
{
    // Ported from Supercluster.KDTree (BoundedPriorityTest), codeandcats/KdTree
    // (NearestNeighbourListTests, PriorityQueueTests) and viliwonka/KDTree (HeapTests)
    public class MinMaxHeapTests
    {
        private static readonly float[] PlanetDistances =
        {
            91700000f, 41400000f, 78300000f, 624400000f, 1250000000f, 2720000000f, 4350000000f
        };

        private static readonly float[] Ages = { 16, 1, 10, 15, 41, 38 };

        [Test]
        public void PushObjMax_WhenFull_KeepsSmallestValues()
        {
            var heap = new MinMaxHeap<int>(3, Allocator.Temp);
            heap.PushObjMax(34, 98744.90383f);
            heap.PushObjMax(23, 67.39030f);
            heap.PushObjMax(2, 2f);
            heap.PushObjMax(89, 3f);

            Assert.AreEqual(23, heap.PopObjMax());
            Assert.AreEqual(89, heap.PopObjMax());
            Assert.AreEqual(2, heap.PopObjMax());
            heap.Dispose();
        }

        [Test]
        public void PushObjMax_MoreThanCapacity_CountStaysAtCapacity()
        {
            var heap = new MinMaxHeap<int>(5, Allocator.Temp);
            for (var i = 0; i < PlanetDistances.Length; i++)
            {
                heap.PushObjMax(i, PlanetDistances[i]);
            }

            Assert.AreEqual(5, heap.Count);
            heap.Dispose();
        }

        [Test]
        public void PopObjMax_WhenFull_ReturnsKeptValuesFarthestFirst()
        {
            var heap = new MinMaxHeap<int>(5, Allocator.Temp);
            for (var i = 0; i < PlanetDistances.Length; i++)
            {
                heap.PushObjMax(i, PlanetDistances[i]);
            }

            var expected = Enumerable.Range(0, PlanetDistances.Length)
                .OrderBy(i => PlanetDistances[i])
                .Take(5)
                .Reverse()
                .ToArray();

            foreach (var index in expected)
            {
                Assert.AreEqual(index, heap.PopObjMax());
            }

            Assert.AreEqual(0, heap.Count);
            heap.Dispose();
        }

        [Test]
        public void PopObjMax_ReturnsDescendingOrder()
        {
            var heap = new MinMaxHeap<int>(Ages.Length, Allocator.Temp);
            for (var i = 0; i < Ages.Length; i++)
            {
                heap.PushObjMax(i, Ages[i]);
            }

            var expected = Enumerable.Range(0, Ages.Length).OrderByDescending(i => Ages[i]).ToArray();
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], heap.PopObjMax());
                Assert.AreEqual(expected.Length - i - 1, heap.Count);
            }

            heap.Dispose();
        }

        [Test]
        public void PopObjMin_ReturnsAscendingOrder()
        {
            var heap = new MinMaxHeap<int>(Ages.Length, Allocator.Temp);
            for (var i = 0; i < Ages.Length; i++)
            {
                heap.PushObjMin(i, Ages[i]);
            }

            var expected = Enumerable.Range(0, Ages.Length).OrderBy(i => Ages[i]).ToArray();
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], heap.PopObjMin());
                Assert.AreEqual(expected.Length - i - 1, heap.Count);
            }

            heap.Dispose();
        }

        [Test]
        public void PushObjMax_RandomValues_KeepsKSmallest()
        {
            const int k = 13;
            const int count = 100;
            var random = new Unity.Mathematics.Random(12345);
            var values = Enumerable.Range(0, count).Select(_ => random.NextFloat()).ToArray();

            var heap = new MinMaxHeap<int>(k, Allocator.Temp);
            for (var i = 0; i < count; i++)
            {
                heap.PushObjMax(i, values[i]);
            }

            var expected = Enumerable.Range(0, count).OrderBy(i => values[i]).Take(k).Reverse().ToArray();
            foreach (var index in expected)
            {
                Assert.AreEqual(index, heap.PopObjMax());
            }

            heap.Dispose();
        }

        [Test]
        public void Resize_KeepsPushedItems()
        {
            var heap = new MinMaxHeap<int>(2, Allocator.Temp);
            heap.PushObjMax(0, 5f);
            heap.PushObjMax(1, 1f);
            heap.Resize(4);
            heap.PushObjMax(2, 3f);
            heap.PushObjMax(3, 4f);

            Assert.AreEqual(4, heap.Count);
            Assert.AreEqual(0, heap.PopObjMax());
            Assert.AreEqual(3, heap.PopObjMax());
            Assert.AreEqual(2, heap.PopObjMax());
            Assert.AreEqual(1, heap.PopObjMax());
            heap.Dispose();
        }
    }
}
