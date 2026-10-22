using Unity.Collections;
using Unity.Mathematics;
using KNN;
using KNN.Jobs;
using Unity.Jobs;
using UnityEngine.Profiling;
using Random = Unity.Mathematics.Random;

public static class KnnApiDemo  {
	public static void Demo() {
		Profiler.BeginSample("Test Query");
		
		// First let's create a random point cloud
		var points = new NativeArray<float3>(100000, Allocator.Persistent);
		var rand = new Random(123456);
		for (int i = 0; i < points.Length; ++i) {
			points[i] = rand.NextFloat3();
		}

		// Number of neighbours we want to query
		const int kNeighbours = 10;
		float3 queryPosition = float3.zero;
		
		Profiler.BeginSample("Build");
		// Create a container that accelerates querying for neighbours.
		// The 2nd argument indicates whether we want to build the tree straight away or not
		// Let's hold off on building it a little bit
		var knnContainer = new KnnContainer(points, false, Allocator.TempJob);
		Profiler.EndSample();
		
		// Whenever your point cloud changes, you can make a job to rebuild the container:
		var rebuildJob = new KnnRebuildJob(knnContainer);
		rebuildJob.Schedule().Complete();
		
		// Most basic usage:
		// Get 10 nearest neighbours as indices into our points array!
		// This is NOT burst accelerated yet! Unity need to implement compiling delegates with Burst
		var result = new NativeArray<int>(kNeighbours, Allocator.TempJob);
		knnContainer.QueryKNearest(queryPosition, result);

		// The result array at this point contains indices into the points array with the nearest neighbours!
		
		Profiler.BeginSample("Simple Query");
		// Get a job to do the query.
		var queryJob = new QueryKNearestJob(knnContainer, queryPosition, result);
		
		// And just run immediately on the main thread for now. This uses Burst!
		queryJob.Schedule().Complete();
		Profiler.EndSample();
		
		// Or maybe we want to query neighbours for multiple points.
		const int queryPoints = 100000;

		// Query at a few random points, each with its own k
		var kInput = new NativeArray<KNearestData>(queryPoints, Allocator.TempJob);
		for (int i = 0; i < queryPoints; ++i) {
			kInput[i] = new KNearestData { Position = rand.NextFloat3() * 0.1f, K = kNeighbours };
		}

		// One stream index per query; each holds the neighbour indices, nearest first
		var kOutput = new NativeStream(kInput.Length, Allocator.TempJob);

		Profiler.BeginSample("Batch Query");
		// Fire up job to get results for all points
		var batchQueryJob = new KnnBatchKNearestJob { Container = knnContainer, Input = kInput, Output = kOutput.AsWriter() };

		// And just run immediately now. This will run on multiple threads!
		batchQueryJob.ScheduleBatch(kInput.Length, kInput.Length / 32).Complete();
		Profiler.EndSample();

		// Or maybe we're interested in a range around eacht query point
		var queryRangeResult = new NativeList<int>(Allocator.TempJob);
		var rangeQueryJob = new QueryRangeJob(knnContainer, queryPosition, 2.0f, queryRangeResult);

		// And just run immediately on the main thread for now. This uses Burst!
		rangeQueryJob.Schedule().Complete();

		// Batch range queries write a variable number of indices per query, so no max count is needed upfront.
		// Keep the radius small: every point in range is written, and the points fill a unit cube
		const float batchRadius = 0.02f;
		var radiusInput = new NativeArray<RadiusData>(queryPoints, Allocator.TempJob);
		for (int i = 0; i < queryPoints; ++i) {
			radiusInput[i] = new RadiusData { Position = kInput[i].Position, Radius = batchRadius };
		}

		var radiusOutput = new NativeStream(radiusInput.Length, Allocator.TempJob);

		Profiler.BeginSample("Batch Range Query");
		// Fire up job to get results for all points
		var batchRange = new KnnBatchRadiusJob { Container = knnContainer, Input = radiusInput, Output = radiusOutput.AsWriter() };

		// And just run immediately now. This will run on multiple threads!
		batchRange.ScheduleBatch(radiusInput.Length, radiusInput.Length / 32).Complete();
		Profiler.EndSample();

		// Now the output streams contain all the neighbours!
		queryRangeResult.Dispose();
		radiusInput.Dispose();
		radiusOutput.Dispose();
		knnContainer.Dispose();
		kInput.Dispose();
		kOutput.Dispose();
		points.Dispose();
		result.Dispose();
		Profiler.EndSample();
	}
}
