  *	�������@2x
AIterator::Root::BatchV2::Prefetch::Shuffle::Zip[0]::ParallelMapV2���o�6@!���L�RL@)���o�6@1���L�RL@:Preprocessing2X
!Iterator::Root::BatchV2::Prefetch��:�#@!�����8@)��:�#@1�����8@:Preprocessing2N
Iterator::Root::BatchV2�C����0@!h&U�g�D@)���B�i@1�ˎ�1@:Preprocessing2f
/Iterator::Root::BatchV2::Prefetch::Shuffle::Zip�~�:p7@!�9?��L@)Zd;�O��?1�@Ժ���?:Preprocessing2�
NIterator::Root::BatchV2::Prefetch::Shuffle::Zip[0]::ParallelMapV2::TensorSliceӼ���?!�:Җ��?)Ӽ���?1�:Җ��?:Preprocessing2a
*Iterator::Root::BatchV2::Prefetch::Shuffle��_vO>7@!�ѐHK�L@)�Zd;߿?1�̈���?:Preprocessing2�
NIterator::Root::BatchV2::Prefetch::Shuffle::Zip[1]::ParallelMapV2::TensorSlice�(���?!�SoY���?)�(���?1�SoY���?:Preprocessing2x
AIterator::Root::BatchV2::Prefetch::Shuffle::Zip[1]::ParallelMapV2z�):�˯?!�1�O��?)z�):�˯?1�1�O��?:Preprocessing2E
Iterator::Root'1��0@!R���D@)Q�|a�?1���%�Ҷ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.