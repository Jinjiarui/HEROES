#!/bin/bash
#source src/global_conf.sh
HADOOP_BIN=/NAS2020/Share/chenxianyu/hadoop/bin/hadoop
HADOOP_STREAMMING=/NAS2020/Share/chenxianyu/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.0.jar

# tr te
if [ $# -eq 1 ]; then
  task_type=$1
else
  # shellcheck disable=SC2209
  task_type=test
fi

echo "remap_sample on ${task_type}/sample ..."

# shellcheck disable=SC2125
INPUT_PATH=${task_type}/sample/part*
OUTPUT_PATH=${task_type}/remap_sample

${HADOOP_BIN} fs -rm -r ${OUTPUT_PATH}/

${HADOOP_BIN} jar ${HADOOP_STREAMMING} \
  -input "${INPUT_PATH}" \
  -output ${OUTPUT_PATH} \
  -mapper "python get_remap_mapper.py /NAS2020/Share/chenxianyu/PycharmProjects/Horse/alicpp/cnts" \
  -reducer "python get_remap_reducer.py" \
  -file "get_remap_mapper.py" \
  -file "get_remap_reducer.py" \
  -jobconf mapreduce.job.priority=HIGH \
  -jobconf mapreduce.map.memory.mb=28192 \
  -jobconf mapreduce.map.java.opts=-Xmx8000m \
  -jobconf mapreduce.reduce.memory.mb=28192 \
  -jobconf mapreduce.reduce.java.opts=-Xmx8000m \
  -jobconf mapred.map.capacity.per.tasktracker=3 \
  -jobconf mapred.reduce.capacity.per.tasktracker=3 \
  -jobconf mapred.task.timeout=7200000 \
  -jobconf mapreduce.job.maps=500 \
  -jobconf mapreduce.job.reduces=1 \
  -jobconf mapreduce.job.queuename=root.mtt.default \
  -jobconf mapreduce.job.name="t_sd_mtt_aliccp_stat_${task_type}_lambdaji"

# shellcheck disable=SC2181
if [ ${?} -eq 0 ]; then
  echo "succeed"
else
  echo "failed"
  exit 1
fi

echo "remap_sample on ${HADOOP_ROOT}/lambdaji/aliccp/${task_type}/sample stat:${?}"
