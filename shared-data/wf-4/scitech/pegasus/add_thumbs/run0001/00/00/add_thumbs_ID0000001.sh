#!/bin/bash
set -e
pegasus_lite_version_major="5"
pegasus_lite_version_minor="0"
pegasus_lite_version_patch="0dev"
pegasus_lite_enforce_strict_wp_check="true"
pegasus_lite_version_allow_wp_auto_download="true"


. pegasus-lite-common.sh

pegasus_lite_init

# cleanup in case of failures
trap pegasus_lite_signal_int INT
trap pegasus_lite_signal_term TERM
trap pegasus_lite_unexpected_exit EXIT

printf "\n########################[Pegasus Lite] Setting up workdir ########################\n"  1>&2
# work dir
export pegasus_lite_work_dir=$PWD
pegasus_lite_setup_work_dir

printf "\n##############[Pegasus Lite] Figuring out the worker package to use ##############\n"  1>&2
# figure out the worker package to use
pegasus_lite_worker_package

printf "\n######[Pegasus Lite] Checking file integrity for transferred container files ######\n"  1>&2
# do file integrity checks
pegasus-integrity --print-timings --verify=stdin 1>&2 << 'eof'
thumbs_up
eof
set -e

printf "\n########[Pegasus Lite] Writing out script to launch user task in container ########\n"  1>&2

cat <<EOF > add_thumbs_ID0000001-cont.sh
#!/bin/bash
set -e
# setting environment variables for job
export PATH=\$root_path
EOF
container_env /scratch >> add_thumbs_ID0000001-cont.sh
cat <<EOF2 >> add_thumbs_ID0000001-cont.sh
pegasus_lite_version_major=$pegasus_lite_version_major
pegasus_lite_version_minor=$pegasus_lite_version_minor
pegasus_lite_version_patch=$pegasus_lite_version_patch
pegasus_lite_enforce_strict_wp_check=false
pegasus_lite_version_allow_wp_auto_download=$pegasus_lite_version_allow_wp_auto_download
pegasus_lite_work_dir=/scratch

. ./pegasus-lite-common.sh
pegasus_lite_init


printf "\n##############[Container] Figuring out Pegasus worker package to use ##############\n"  1>&2
# figure out the worker package to use
pegasus_lite_worker_package
echo "PATH in container is set to is set to \$PATH"  1>&2

printf "\n###############[Container] Checking file integrity for input files ###############\n"  1>&2
# do file integrity checks
pegasus-integrity --print-timings --verify=stdin 1>&2 << 'eof'
input.txt:thumbs_up
eof

printf "\n#########################[Container] Launching user task #########################\n"  1>&2

pegasus-kickstart  -n add_thumbs -N ID0000001 -R condorpool  -s output.txt=output.txt -L add_thumbs -T 2020-07-09T23:35:41+00:00 /usr/bin/add_thumbs_up.py input.txt output.txt
set -e
EOF2


chmod +x add_thumbs_ID0000001-cont.sh
if ! [ $pegasus_lite_start_dir -ef . ]; then
	cp $pegasus_lite_start_dir/pegasus-lite-common.sh . 
fi

set +e
job_ec=0
docker_init thumbs_up
job_ec=$(($job_ec + $?))

docker run --user root -v $PWD:/scratch -w=/scratch --entrypoint /bin/sh --name $cont_name  $cont_image -c "set -e ;export root_path=\$PATH ;if ! grep -q -E  "^$cont_group:" /etc/group ; then groupadd --gid $cont_groupid $cont_group ;fi; if ! id $cont_user 2>/dev/null >/dev/null; then useradd --uid $cont_userid --gid $cont_groupid $cont_user; fi; su $cont_user -c \"./add_thumbs_ID0000001-cont.sh \""
job_ec=$(($job_ec + $?))

docker rm $cont_name  1>&2
job_ec=$(($job_ec + $?))


set -e


# clear the trap, and exit cleanly
trap - EXIT
pegasus_lite_final_exit

