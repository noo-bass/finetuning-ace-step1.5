import type { Clip, Ratings, RoundSummary } from './types'

async function toJson<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export function fetchRounds(): Promise<RoundSummary[]> {
  return fetch('/api/rounds').then((r) => toJson<RoundSummary[]>(r))
}

export function fetchRound(name: string): Promise<{ name: string; clips: Clip[] }> {
  return fetch(`/api/rounds/${encodeURIComponent(name)}`).then((r) =>
    toJson<{ name: string; clips: Clip[] }>(r),
  )
}

export async function fetchRatings(name: string): Promise<Ratings> {
  const doc = await fetch(`/api/rounds/${encodeURIComponent(name)}/ratings`).then((r) =>
    toJson<{ ratings?: Ratings }>(r),
  )
  return doc.ratings ?? {}
}

export function putRatings(name: string, ratings: Ratings): Promise<unknown> {
  return fetch(`/api/rounds/${encodeURIComponent(name)}/ratings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ratings }),
  }).then((r) => toJson(r))
}
