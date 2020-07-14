import logging
from pathlib import Path
from Pegasus.api import *

logging.basicConfig(level = logging.DEBUG)

fa = File("input.txt")
rc = ReplicaCatalog()
rc.add_replica("local", fa, Path(".").resolve() / "input.txt")

tc = TransformationCatalog()

thumbs_up = Container(
                "thumbs_up",
                Container.DOCKER,
                image="docker://vedularaghu/thumbs_up:latest"
            )

tc.add_containers(thumbs_up)

add_thumbs = Transformation(
                        "add_thumbs",
                        site="condorpool",
                        pfn="/usr/bin/add_thumbs_up.py",
                        is_stageable=False,
                        container=thumbs_up
                )

tc.add_transformations(add_thumbs)

wf = Workflow("add_thumbs")

fb = File("output.txt")

job_add_thumbs = Job(add_thumbs)\
                    .add_args(fa, fb)\
                    .add_inputs(fa)\
                    .add_outputs(fb)

wf.add_jobs(job_add_thumbs)
wf.add_replica_catalog(rc)
wf.add_transformation_catalog(tc)

try:
    wf.plan(submit=True)\
        .wait()\
        .analyze()\
        .statistics()
except PegasusClientError as e:
    print(e.output)
