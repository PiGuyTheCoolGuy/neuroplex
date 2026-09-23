"""Vectorized square-block geometry; all coordinates are physical world units."""

import numpy as np


def ray_hits(origins, directions, blocks, half):
    origins = np.atleast_2d(origins)
    directions = np.atleast_2d(directions)
    if not len(blocks):
        return np.full((len(origins), 0), np.inf)
    delta = blocks[None, :, :] - origins[:, None, :]
    parallel = np.abs(directions[:, None, :]) < 1e-10
    safe = np.where(parallel, 1.0, directions[:, None, :])
    a, b = (delta - half) / safe, (delta + half) / safe
    low = np.where(parallel, -np.inf, np.minimum(a, b))
    high = np.where(parallel, np.inf, np.maximum(a, b))
    near, far = low.max(axis=2), high.min(axis=2)
    valid = (far >= np.maximum(near, 0)) & ~np.any(parallel & (np.abs(delta) > half), axis=2)
    return np.where(valid, np.maximum(near, 0), np.inf)


def circle_overlaps(point, radius, blocks, half):
    offset = np.maximum(np.abs(blocks - point) - half, 0)
    return np.sum(offset * offset, axis=-1) < radius * radius - 1e-9


def occluded(origin, points, blocks, half):
    if not len(points):
        return np.zeros(0, dtype=bool)
    directions = points - origin
    hits = ray_hits(np.repeat([origin], len(points), axis=0), directions, blocks, half)
    return np.any(hits < 1 - 1e-7, axis=1)


def cover_at(points, blocks, half, radius, width, height):
    """Fraction of 16 short sight lines shielded by >=2 separate blocks.

    Reject body overlaps and require adjacent open clearance rays (an exit).
    World edges never count as construction. This is an explicit geometric
    proxy for useful cover, not a blueprint or proof of a complete shelter.
    """
    points = np.atleast_2d(points)
    if len(blocks) < 2:
        return np.zeros(len(points))
    angles = np.arange(16) * 2 * np.pi / 16
    directions = np.tile(np.column_stack((np.cos(angles), np.sin(angles))), (len(points), 1))
    origins = np.repeat(points, 16, axis=0)
    sight = ray_hits(origins, directions, blocks, half).reshape(len(points), 16, len(blocks))
    clearance = ray_hits(origins, directions, blocks, half + radius).min(axis=1).reshape(len(points), 16)
    # Keep exits inside the world too; borders are obstacles, never cover credit.
    exits = origins + directions * 6
    inside = ((exits >= radius) & (exits <= [width - radius, height - radius])).all(axis=1).reshape(len(points), 16)
    open_rays = (clearance > 6) & inside
    has_exit = (open_rays & np.roll(open_rays, 1, axis=1)).any(axis=1)
    closest = sight.argmin(axis=2)
    blocked = sight.min(axis=2) < 8
    first = np.where(blocked, closest, -1).max(axis=1)
    multiple = (blocked & (closest != first[:, None])).any(axis=1)
    offsets = np.maximum(np.abs(points[:, None, :] - blocks[None, :, :]) - half, 0)
    valid = (np.sum(offsets * offsets, axis=2) >= radius * radius).all(axis=1)
    valid &= ((points >= radius) & (points <= [width - radius, height - radius])).all(axis=1)
    return blocked.mean(axis=1) * has_exit * multiple * valid
