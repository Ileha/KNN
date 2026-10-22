using KNN;
using KNN.Jobs;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.ParticleSystemJobs;

// Ideally you really would use something like ECS to visualize the point cloud
// And query from a job component system
// But this is meant to be a really simple demo and not an ECS demo
// I might look into deeper ECS integration at some point
public class KnnVisualizationDemo : MonoBehaviour {
	public enum QueryMode {
		KNearest,
		Range
	}

	public QueryMode Mode;

	public int ParticleCount = 20000;

	public int QueryK = 20;
	public float QueryRange = 1.0f;

	ParticleSystem m_system;

	NativeArray<KNearestData> m_kNearestInput;
	NativeArray<RadiusData> m_radiusInput;
	NativeArray<Color32> m_queryColors;

	NativeArray<float3> m_points;

	// One stream index per probe, holding the indices of its neighbours
	NativeStream m_results;

	KnnContainer m_container;

	JobHandle m_particleJobHandle;

	void Start() {
		m_system = GetComponent<ParticleSystem>();

		m_system.Emit(ParticleCount);
		m_points = new NativeArray<float3>(ParticleCount, Allocator.Persistent);

		// Create a container that accelerates querying for neighbours
		m_container = new KnnContainer(m_points, false, Allocator.Persistent); // Skip building for now. We rebuild every frame
	}

	void OnDestroy() {
		m_particleJobHandle.Complete();
		m_points.Dispose();
		m_container.Dispose();
		DisposeResults();

		// Query buffers are created in the first LateUpdate
		if (m_queryColors.IsCreated) {
			m_kNearestInput.Dispose();
			m_radiusInput.Dispose();
			m_queryColors.Dispose();
		}
	}

	// Both query modes write one stream index per probe, so one job colors them
	struct ParticleJob : IJobParticleSystem {
		[ReadOnly] public NativeStream.Reader KnnResults;
		public NativeArray<float3> Points;
		public NativeArray<Color32> Colors;

		public void Execute(ParticleSystemJobData jobData) {
			var colors = jobData.startColors;
			var positions = jobData.positions;

			for (int i = 0; i < jobData.count; ++i) {
				Points[i] = new float3(positions.x[i], positions.y[i], positions.z[i]);
			}

			// Set every particle to white first
			for (int i = 0; i < jobData.count; i++) {
				colors[i] = new Color32(0, 0, 0, 255);
			}

			// Set all neighbours to result color
			for (int i = 0; i < Colors.Length; i++) {
				int count = KnnResults.BeginForEachIndex(i);
				for (int j = 0; j < count; j++) {
					colors[KnnResults.Read<int>()] = Colors[i];
				}

				KnnResults.EndForEachIndex();
			}
		}
	}

	// Unity calls this when it schedules the particle update jobs, replacing the removed ParticleSystem.SetJob
	void OnParticleUpdateJobScheduled() {
		// Query results are created in the first LateUpdate
		if (!m_results.IsCreated) {
			return;
		}

		// Update particles job to do the colors
		m_particleJobHandle = new ParticleJob {
			KnnResults = m_results.AsReader(),
			Points = m_points,
			Colors = m_queryColors
		}.Schedule(m_system);
	}

	// After particle job
	void LateUpdate() {
		// Particle jobs run until PostLateUpdate, and they write m_points that the rebuild reads
		m_particleJobHandle.Complete();

		// Rebuild our datastructure
		var rebuild = new KnnRebuildJob(m_container);
		var rebuildHandle = rebuild.Schedule();

		// Get all probe positions / colors
		if (!m_queryColors.IsCreated || m_queryColors.Length != QueryProbe.All.Count) {
			if (m_queryColors.IsCreated) {
				m_kNearestInput.Dispose();
				m_radiusInput.Dispose();
				m_queryColors.Dispose();
			}

			m_kNearestInput = new NativeArray<KNearestData>(QueryProbe.All.Count, Allocator.Persistent);
			m_radiusInput = new NativeArray<RadiusData>(QueryProbe.All.Count, Allocator.Persistent);
			m_queryColors = new NativeArray<Color32>(QueryProbe.All.Count, Allocator.Persistent);
		}

		for (int i = 0; i < QueryProbe.All.Count; i++) {
			var p = QueryProbe.All[i];
			m_kNearestInput[i] = new KNearestData { Position = p.transform.position, K = QueryK };
			m_radiusInput[i] = new RadiusData { Position = p.transform.position, Radius = QueryRange };
			m_queryColors[i] = p.Color;
		}

		// A stream cannot be cleared, so each frame writes a new one
		DisposeResults();
		m_results = new NativeStream(QueryProbe.All.Count, Allocator.Persistent);

		// We're only doing a very limited number of points - so allow each query to have it's own job
		switch (Mode) {
			case QueryMode.KNearest: {
				// Do a KNN query, dependent on the rebuild
				var query = new KnnBatchKNearestJob { Container = m_container, Input = m_kNearestInput, Output = m_results.AsWriter() };
				query.ScheduleBatch(m_kNearestInput.Length, 1, rebuildHandle).Complete();
				break;
			}

			case QueryMode.Range: {
				// Do a range query, dependent on the rebuild
				var query = new KnnBatchRadiusJob { Container = m_container, Input = m_radiusInput, Output = m_results.AsWriter() };
				query.ScheduleBatch(m_radiusInput.Length, 1, rebuildHandle).Complete();
				break;
			}
		}
	}

	void DisposeResults() {
		if (m_results.IsCreated) {
			m_results.Dispose();
		}
	}
}