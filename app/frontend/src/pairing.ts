import type { Clip, Comparison } from './types'

/** Elo-style pairing engine for compare mode.
 *
 * Everything here is a pure function of (clips, comparison history), so the
 * engine is deterministic: replaying the same history always yields the same
 * scores, the same next pair, and the same ranking. Undo is just "pop the
 * last comparison".
 */

const K = 32
const BASE_SCORE = 1000

/** Target comparison count for n clips before we call the ranking stable. */
export function targetComparisons(clipCount: number): number {
  return clipCount * 2
}

export function computeScores(
  clips: Clip[],
  comparisons: Comparison[],
): Map<string, number> {
  const scores = new Map(clips.map((c) => [c.id, BASE_SCORE]))
  for (const cmp of comparisons) {
    const ra = scores.get(cmp.a)
    const rb = scores.get(cmp.b)
    if (ra === undefined || rb === undefined) continue
    const expectedA = 1 / (1 + 10 ** ((rb - ra) / 400))
    const scoreA = cmp.winner === cmp.a ? 1 : 0
    scores.set(cmp.a, ra + K * (scoreA - expectedA))
    scores.set(cmp.b, rb + K * (1 - scoreA - (1 - expectedA)))
  }
  return scores
}

export function comparisonCounts(
  clips: Clip[],
  comparisons: Comparison[],
): Map<string, number> {
  const counts = new Map(clips.map((c) => [c.id, 0]))
  for (const cmp of comparisons) {
    for (const id of [cmp.a, cmp.b]) {
      const n = counts.get(id)
      if (n !== undefined) counts.set(id, n + 1)
    }
  }
  return counts
}

/** Clip ids ranked best-first by current Elo score (ties broken by id). */
export function rankClips(clips: Clip[], comparisons: Comparison[]): string[] {
  const scores = computeScores(clips, comparisons)
  return [...clips]
    .sort(
      (x, y) =>
        scores.get(y.id)! - scores.get(x.id)! || x.id.localeCompare(y.id),
    )
    .map((c) => c.id)
}

function pairKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`
}

/** Adaptive next pair: anchor on the clip with the fewest comparisons (ties
 * by id), then partner it with the closest-scored clip, preferring pairs not
 * yet played (ties by fewer comparisons, then id). Returns ids [a, b]. */
export function nextPair(
  clips: Clip[],
  comparisons: Comparison[],
): [string, string] | null {
  if (clips.length < 2) return null
  const scores = computeScores(clips, comparisons)
  const counts = comparisonCounts(clips, comparisons)
  const played = new Set(comparisons.map((c) => pairKey(c.a, c.b)))

  const anchor = [...clips].sort(
    (x, y) => counts.get(x.id)! - counts.get(y.id)! || x.id.localeCompare(y.id),
  )[0]

  const pickClosest = (pool: Clip[]): Clip =>
    [...pool].sort(
      (x, y) =>
        Math.abs(scores.get(x.id)! - scores.get(anchor.id)!) -
          Math.abs(scores.get(y.id)! - scores.get(anchor.id)!) ||
        counts.get(x.id)! - counts.get(y.id)! ||
        x.id.localeCompare(y.id),
    )[0]

  const others = clips.filter((c) => c.id !== anchor.id)
  const fresh = others.filter((c) => !played.has(pairKey(anchor.id, c.id)))
  const partner = pickClosest(fresh.length > 0 ? fresh : others)
  return [anchor.id, partner.id]
}
