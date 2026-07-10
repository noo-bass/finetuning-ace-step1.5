export type Tier = 'top' | 'keep' | 'pass'

export type Mode = 'tiers' | 'pairwise'

export interface RoundSummary {
  name: string
  clipCount: number
  rated: boolean
}

export interface Clip {
  id: string
  url: string
}

export interface Rating {
  tier?: Tier
  artifact?: boolean
  ts?: string
}

export type Ratings = Record<string, Rating>

export interface Comparison {
  a: string
  b: string
  winner: string
  ts?: string
}

/** The persisted ratings document. Tier-era docs carry no mode/comparisons;
 * they are read as mode "tiers" with an empty comparison history. */
export interface RatingsDoc {
  mode: Mode
  ratings: Ratings
  comparisons: Comparison[]
}

export type SaveState = 'clean' | 'dirty' | 'saving' | 'error'
