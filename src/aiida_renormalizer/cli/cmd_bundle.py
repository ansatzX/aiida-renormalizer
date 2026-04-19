# -*- coding: utf-8 -*-
"""verdi reno bundle command."""
from __future__ import annotations

import click
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.cmdline.utils import echo
from aiida.orm import load_node

from aiida_renormalizer.data.artifacts import export_publication_bundle


@click.command("bundle")
@click.option(
    "-n",
    "--node",
    "node_pk",
    type=int,
    required=True,
    help="Node ID (PK) of the artifact-backed data node",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Bundle output directory",
)
@click.option(
    "--relink",
    is_flag=True,
    default=False,
    help="Relink the node to the bundled artifact after export",
)
@with_dbenv()
def bundle(node_pk: int, output_dir: str, relink: bool) -> None:
    """Export stored artifact nodes to a publication bundle."""
    try:
        node = load_node(node_pk)
    except Exception as exc:
        echo.echo_critical(f"Failed to load node {node_pk}: {exc}")

    if not hasattr(node, "artifact_metadata"):
        echo.echo_critical(
            f"Node {node_pk} does not expose artifact metadata and cannot be bundled."
        )

    manifest = export_publication_bundle(
        node_uuid=str(node.uuid),
        artifact=node.artifact_metadata,
        summary={
            key: value
            for key, value in node.base.attributes.all.items()
            if key
            not in {
                "storage_backend",
                "storage_base",
                "relative_path",
                "artifact_format",
                "artifact_size",
                "content_hash",
            }
        },
        output_dir=output_dir,
    )

    if relink:
        if not hasattr(node, "relink_artifact"):
            echo.echo_critical(
                f"Node {node_pk} cannot be relinked after bundle export."
            )
        node.relink_artifact(output_dir, manifest["bundle_relative_path"])

    echo.echo_success(f"Bundle written to: {output_dir}")
    echo.echo_info(f"Manifest: {manifest['bundle_relative_path']}")
