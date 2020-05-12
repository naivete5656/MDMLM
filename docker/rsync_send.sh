#!/bin/bash

rsync -arv /media/kazuya/ボリューム/Main_Disk1/ uchida_labmem@192.168.1.38:/data/MainDisk/server1data

set -eu


MESSAGEFILE="finish rsync"

trap "rm ${MESSAGEFILE}" 0

MESSAGE="finish rsync"

# WebHookのURL
URL='https://hooks.slack.com/services/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'
# 送信先のチャンネル
CHANNEL=${CHANNEL:-'#bot_channel'}
# botの名前
BOTNAME=${BOTNAME:-'rsync-bot'}
# 絵文字
EMOJI=${EMOJI:-':sushi:'}
# 見出し
HEAD=${HEAD:-"[hddを交換してください]\n"}

# メッセージをシンタックスハイライト付きで取得
MESSAGE='```'`cat ${MESSAGEFILE}`'```'

# json形式に整形
payload="payload={
    \"channel\": \"${CHANNEL}\",
    \"username\": \"${BOTNAME}\",
    \"icon_emoji\": \"${EMOJI}\",
    \"text\": \"${HEAD}${MESSAGE}\"
}"

# 送信
curl -s -S -X POST --data-urlencode "${payload}" ${URL} >/dev/null
