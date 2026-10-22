![Demo Image](http://g2f.nl/0jx82rw)

# KNN

K-nearest neighbour, radius and box queries for Unity on a k-d tree. Runs in Burst jobs, no managed allocations. Based on [viliwonka/KDTree](https://github.com/viliwonka/KDTree).

## Build

```C#
var points = new NativeArray<float3>(count, Allocator.Persistent);
var knn = new KnnContainer(points, buildNow: false, Allocator.Persistent);

// After the points move, or after assigning a new array to knn.Points
new KnnRebuildJob(knn).Schedule().Complete();

knn.Dispose();
points.Dispose();
```

## Single queries

Results are indices into `points`.

```C#
// k nearest, farthest first. k must be in 1..points.Length
var nearest = new NativeArray<int>(k, Allocator.TempJob);
new QueryKNearestJob(knn, position, nearest).Run();

// All points within radius, unsorted. Adds to the list
var inRange = new NativeList<int>(Allocator.TempJob);
new QueryRangeJob(knn, position, radius, inRange).Run();

// All points inside the box, bounds included, unsorted. Adds to the list
var inBox = new NativeList<int>(Allocator.TempJob);
new QueryIntervalJob { Container = knn, Min = min, Max = max, Result = inBox }.Run();

// The same calls without a job (no Burst):
knn.QueryKNearest(position, nearest);
knn.QueryRange(position, radius, inRange);
knn.QueryInterval(min, max, inBox);
```

## Batch queries

One input per query, one `NativeStream` index per query in the output.

```C#
var input = new NativeArray<KNearestData>(queryCount, Allocator.TempJob);
input[0] = new KNearestData { Position = position, K = 10 };   // k is clamped to the point count
var output = new NativeStream(input.Length, Allocator.TempJob);

new KnnBatchKNearestJob { Container = knn, Input = input, Output = output.AsWriter() }
    .ScheduleBatch(input.Length, 32).Complete();

// Same shape, unsorted results:
// KnnBatchRadiusJob   + RadiusData   { Position, Radius }
// KnnBatchIntervalJob + IntervalData { Min, Max }

var reader = output.AsReader();
for (int q = 0; q < input.Length; ++q) {
    int count = reader.BeginForEachIndex(q);   // nearest first for KnnBatchKNearestJob
    for (int i = 0; i < count; ++i) {
        int pointIndex = reader.Read<int>();
    }
    reader.EndForEachIndex();
}

input.Dispose();
output.Dispose();
```

## Demo

- `Demo/KnnApiDemo.cs`: the calls above on 100k points.
- `Demo/KnnVisualizationDemo.cs` (`Demo/demo.unity`): particles colored by k-nearest or radius queries around probes, with the tree rebuilt every frame.

## Installation

Add the git URL to your package manifest.

## Third-party notices

The tree build and the queries contain code ported from these projects:

- [viliwonka/KDTree](https://github.com/viliwonka/KDTree): range and interval queries, split rule. MIT License, Copyright (c) 2018 Vili Volčini.
- [scipy](https://github.com/scipy/scipy) (`scipy/spatial/ckdtree`): handling of points that share a coordinate in the build, and taking whole nodes inside the query radius. BSD 3-Clause License.

The unit tests port test cases and data from [ericreg/Supercluster.KDTree](https://github.com/ericreg/Supercluster.KDTree) (MIT), [codeandcats/KdTree](https://github.com/codeandcats/KdTree) (MIT), [viliwonka/KDTree](https://github.com/viliwonka/KDTree) (MIT), [scipy](https://github.com/scipy/scipy) (BSD 3-Clause) and [KristofferC/NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) (MIT).

Full license texts are in [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).
