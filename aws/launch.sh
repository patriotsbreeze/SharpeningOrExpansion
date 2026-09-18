#!/usr/bin/env bash
# Request a spot instance with an on-demand fallback.
#
# p5.48xlarge spot is the right target: 8xH100 80GB at roughly $30/hr against ~$55 on demand,
# with reported interruption under 5%. p4d spot is cheaper but interrupts 15-20% of the time,
# and p4d is A100 *40GB*, not 80GB -- p4de is the 80GB SKU and is usually capacity starved.
# The resume design makes spot safe; it does not make a 20% interruption rate pleasant.
set -Eeuo pipefail

INSTANCE_TYPE="${INSTANCE_TYPE:-p5.48xlarge}"
REGION="${REGION:-us-east-1}"
AMI_ID="${AMI_ID:?set AMI_ID to a Deep Learning AMI (GPU, Ubuntu 22.04) in ${REGION}}"
KEY_NAME="${KEY_NAME:?set KEY_NAME}"
SUBNET_ID="${SUBNET_ID:?set SUBNET_ID}"
SG_ID="${SG_ID:?set SG_ID}"
IAM_PROFILE="${IAM_PROFILE:-soe-node}"   # needs s3:GetObject/PutObject/ListBucket on the bucket
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/soe}"
DISK_GB="${DISK_GB:-500}"
MODE="${MODE:-spot}"

USER_DATA=$(mktemp)
{
  echo '#!/usr/bin/env bash'
  echo "export S3_URI='${S3_URI}'"
  cat "$(dirname "$0")/bootstrap.sh"
} > "${USER_DATA}"

common=(
  --region "${REGION}" --image-id "${AMI_ID}" --instance-type "${INSTANCE_TYPE}"
  --key-name "${KEY_NAME}" --subnet-id "${SUBNET_ID}" --security-group-ids "${SG_ID}"
  --iam-instance-profile "Name=${IAM_PROFILE}"
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${DISK_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
  --user-data "file://${USER_DATA}"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=soe-${INSTANCE_TYPE}},{Key=Project,Value=SharpeningOrExpansion}]"
)

launch() {
  if [[ "$1" == "spot" ]]; then
    aws ec2 run-instances "${common[@]}" \
      --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
      --query 'Instances[0].InstanceId' --output text
  else
    aws ec2 run-instances "${common[@]}" --query 'Instances[0].InstanceId' --output text
  fi
}

echo "[launch] requesting ${MODE} ${INSTANCE_TYPE} in ${REGION}"
if ! ID=$(launch "${MODE}" 2>/tmp/soe_launch_err); then
  cat /tmp/soe_launch_err
  if [[ "${MODE}" == "spot" ]]; then
    echo "[launch] spot unavailable; falling back to on-demand (this costs ~1.8x)"
    ID=$(launch ondemand)
  else
    exit 1
  fi
fi

echo "[launch] instance ${ID}; waiting for running state"
aws ec2 wait instance-running --region "${REGION}" --instance-ids "${ID}"
IP=$(aws ec2 describe-instances --region "${REGION}" --instance-ids "${ID}" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
cat <<MSG

  instance : ${ID}
  address  : ${IP}
  bootstrap: ssh ubuntu@${IP} 'tail -f /var/log/cloud-init-output.log'
  run      : ssh ubuntu@${IP} 'cd /opt/soe && S3_URI=${S3_URI} scripts/launch_node.sh configs/experiments/stage1_tierA.yaml'
  TERMINATE WHEN DONE: aws ec2 terminate-instances --region ${REGION} --instance-ids ${ID}
MSG
rm -f "${USER_DATA}"
