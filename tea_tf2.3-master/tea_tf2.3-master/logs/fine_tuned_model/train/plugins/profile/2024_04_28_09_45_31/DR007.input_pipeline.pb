  *	������ A2x
AIterator::Root::BatchV2::Prefetch::Shuffle::Zip[0]::ParallelMapV2�@���,\@!�&y%>T@)�@���,\@1�&y%>T@:Preprocessing2X
!Iterator::Root::BatchV2::Prefetch��j+�'6@!���<�/@)��j+�'6@1���<�/@:Preprocessing2N
Iterator::Root::BatchV2t$����9@!eJ���2@)��a�ִ@1Q�ʯ�W@:Preprocessing2f
/Iterator::Root::BatchV2::Prefetch::Shuffle::Zip�i�q�B\@!H����MT@)3ı.n��?1����`X�?:Preprocessing2a
*Iterator::Root::BatchV2::Prefetch::Shufflen��J\@!t�]�'ST@)��(��?1��<}J��?:Preprocessing2�
NIterator::Root::BatchV2::Prefetch::Shuffle::Zip[1]::ParallelMapV2::TensorSlice+�����?!GXe+U��?)+�����?1GXe+U��?:Preprocessing2�
NIterator::Root::BatchV2::Prefetch::Shuffle::Zip[0]::ParallelMapV2::TensorSlice�HP��?!�$S�&q�?)�HP��?1�$S�&q�?:Preprocessing2x
AIterator::Root::BatchV2::Prefetch::Shuffle::Zip[1]::ParallelMapV2?�ܵ�|�?!8�����?)?�ܵ�|�?18�����?:Preprocessing2E
Iterator::Root�K7�A�9@!񽍄O�2@)F%u�{?1��8�9ls?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.