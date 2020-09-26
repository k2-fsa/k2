#include "k2/csrc/ragged.h"

namespace k2 {

// NOTE that src is sorted in-place.
template <typename T, typename Op = LessThan<T>>
void SortSublists2(Ragged<T> &src, Array1<int32_t> *order) {
  Array1<int32_t> &segment = src.shape.RowSplits(src.NumAxes() - 1);
  K2_CHECK_EQ(src.values.Dim(), order->Dim());
  // also check src and order are on the same device.

  // TODO(fangjun): create a ModernGPUContext.
  mgpu::standard_context_t context;

  Array1<T> saved = src.values.To(GetCpuContext());

  mgpu::segmented_sort_indices(src.values.Data(), order->Data(),
                               src.values.Dim(), segment.Data() + 1,
                               segment.Dim() - 2, Op(), context);

}

}  // namespace k2

int main() {
  using T = float;
  using namespace k2;
  // constructed with row_splits and row_ids
  // RaggedTensor4 t = [
  //  [ [[ 2, 1], [4]],  [[3, 0]] ],
  //  [ [[9, 7, 8]], [[6], [3, 5, 7]], [[2]] ],
  //  [ [[3, 4], [], [8]] ]
  // ]
  const std::vector<int32_t> row_splits1 = {0, 2, 5, 6};
  const std::vector<int32_t> row_ids1 = {0, 0, 1, 1, 1, 2};
  const std::vector<int32_t> row_splits2 = {0, 2, 3, 4, 6, 7, 10};
  const std::vector<int32_t> row_ids2 = {0, 0, 1, 2, 3, 3, 4, 5, 5, 5};
  const std::vector<int32_t> row_splits3 = {0,  2,  3,  5,  8, 9,
                                            12, 13, 15, 15, 16};
  const std::vector<int32_t> row_ids3 = {0, 0, 1, 2, 2, 3, 3, 3,
                                         4, 5, 5, 5, 6, 7, 7, 9};
  const std::vector<T> values_vec = {2, 1, 4, 3, 0, 9, 7, 8,
                                     6, 3, 5, 7, 2, 3, 4, 8};
  ContextPtr context = GetCudaContext();  // will use to copy data
  std::vector<RaggedShapeDim> axes;
  axes.emplace_back(RaggedShapeDim{Array1<int32_t>(context, row_splits1),
                                   Array1<int32_t>(context, row_ids1),
                                   static_cast<int32_t>(row_ids1.size())});
  axes.emplace_back(RaggedShapeDim{Array1<int32_t>(context, row_splits2),
                                   Array1<int32_t>(context, row_ids2),
                                   static_cast<int32_t>(row_ids2.size())});
  axes.emplace_back(RaggedShapeDim{Array1<int32_t>(context, row_splits3),
                                   Array1<int32_t>(context, row_ids3),
                                   static_cast<int32_t>(row_ids3.size())});

  RaggedShape shape(axes, true);
  Array1<T> values(context, values_vec);
  Ragged<T> ragged(shape, values);
  std::cout << ragged.values;

  Array1<T> saved = ragged.values.To(GetCpuContext());

  Array1<int32_t> order(ragged.Context(), ragged.values.Dim());
  SortSublists(&ragged, &order);

  Array1<int32_t> &segment = ragged.shape.RowSplits(ragged.NumAxes() - 1);
  std::cout << "segment: " << segment << "\n";
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
