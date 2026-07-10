export type Tier = 'top' | 'keep' | 'pass'

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

export type SaveState = 'clean' | 'dirty' | 'saving' | 'error'
