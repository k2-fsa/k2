#include "k2/csrc/array_ops.h"
#include "k2/csrc/ragged.h"

int main() {
  using T = float;
  using namespace k2;

  auto cpu_context = GetCpuContext();
  auto cuda_context = GetCudaContext();

  RaggedShape shape = RandomRaggedShape(false,  // set_row_ids = false,
                                        2,      // min_num_axes = 2,
                                        4,      // max_num_axes = 4,
                                        10,     // min_num_elements = 0,
                                        30);    // max_num_elements = 2000);

  Array1<T> values =
      RandUniformArray1<T>(shape.Context(), shape.NumElements(), 0, 10);
  Ragged<T> ragged(shape, values);
  ragged = ragged.To(cuda_context);

  Array1<int32_t> &segment = ragged.shape.RowSplits(ragged.NumAxes() - 1);
  std::cout << "segment: " << segment << "\n";

  values = values.To(cpu_context);

  std::cout << "unsorted: " << ragged.values << "\n";

  Array1<T> saved = ragged.values.To(GetCpuContext());

  Array1<int32_t> order(ragged.Context(), ragged.values.Dim());
  SortSublists(&ragged, &order);

  std::cout << "\n order " << order << "\n";
  std::cout << "\n saved: " << saved << "\n";
  std::cout << "\n ragged sorted: " << ragged.values << "\n";
  for (int i = 0; i < order.Dim(); ++i) {
    std::cout << saved[order[i]] << ", ";
  }
  std::cout << "\n";
}

#if 0
using namespace mgpu;

std::vector<int> cpu_segsort(const std::vector<int>& data,
  const std::vector<int>& segments) {

  std::vector<int> copy = data;
  int cur = 0;
  for(int seg = 0; seg < segments.size(); ++seg) {
    int next = segments[seg];
    std::sort(copy.data() + cur, copy.data() + next);
    cur = next;
  }
  std::sort(copy.data() + cur, copy.data() + data.size());
  return copy;
}

int main(int argc, char** argv) {
  standard_context_t context;

  for(int count = 1000; count < 1000 + 100; count += count / 10) {

    for(int it = 1; it <= 10; ++it) {

      int num_segments = div_up(count, 100);
      mem_t<int> segs = fill_random(0, count - 1, num_segments, true, context);
      std::vector<int> segs_host = from_mem(segs);
      mem_t<int> data = fill_random(0, 100000, count, false, context);
      mem_t<int> values(count, context);
      std::vector<int> host_data = from_mem(data);

      segmented_sort_indices(data.data(), values.data(), count, segs.data(), 
        num_segments, less_t<int>(), context);

      std::vector<int> ref = cpu_segsort(host_data, segs_host);
      std::vector<int> sorted = from_mem(data);

      // Check that the indices are correct.
      std::vector<int> host_indices = from_mem(values);
      for(int i = 0; i < count; ++i) {
        if(sorted[i] != host_data[host_indices[i]]) {
          printf("count = %8d it = %3d KEY FAILURE\n", count, it);
          exit(0);
        }
      }

      // Check that the keys are sorted.
      bool success = ref == sorted;
      printf("count = %8d it = %3d %s\n", count, it, 
        (ref == sorted) ? "SUCCESS" : "FAILURE");
      if(!success) exit(0);
    }
  }

  return 0;
}

#endif
