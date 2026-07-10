import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchRatings, fetchRound, putRatings } from './api'
import { ClipCard } from './ClipCard'
import { CompareView } from './CompareView'
import {
  IconAlert,
  IconCircleCheck,
  IconDot,
  IconLoader,
  IconRows,
  IconScale,
} from './icons'
import { rankClips, targetComparisons } from './pairing'
import type { Clip, Comparison, Mode, Ratings, SaveState, Tier } from './types'

const SAVE_DEBOUNCE_MS = 800

interface Props {
  roundName: string
  onSaved: () => void
}

export function RoundScreen({ roundName, onSaved }: Props) {
  const [clips, setClips] = useState<Clip[] | null>(null)
  const [mode, setMode] = useState<Mode>('tiers')
  const [ratings, setRatings] = useState<Ratings>({})
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [saveState, setSaveState] = useState<SaveState>('clean')
  const [current, setCurrent] = useState(0)

  const audioRefs = useRef(new Map<string, HTMLAudioElement>())
  const modeRef = useRef<Mode>('tiers')
  const ratingsRef = useRef<Ratings>({})
  const comparisonsRef = useRef<Comparison[]>([])
  const clipsRef = useRef<Clip[]>([])
  const currentRef = useRef(0)
  modeRef.current = mode
  ratingsRef.current = ratings
  comparisonsRef.current = comparisons
  clipsRef.current = clips ?? []
  currentRef.current = current

  // Load clips and any previously saved ratings doc.
  useEffect(() => {
    let cancelled = false
    Promise.all([fetchRound(roundName), fetchRatings(roundName)]).then(
      ([round, doc]) => {
        if (cancelled) return
        setClips(round.clips)
        setMode(doc.mode)
        setRatings(doc.ratings)
        setComparisons(doc.comparisons)
        setSaveState('clean')
        setCurrent(0)
      },
    )
    return () => {
      cancelled = true
    }
  }, [roundName])

  const rate = useCallback((id: string, action: Tier | 'flag') => {
    setRatings((prev) => {
      const r = { ...(prev[id] ?? {}) }
      if (action === 'flag') r.artifact = !r.artifact
      else r.tier = action
      r.ts = new Date().toISOString()
      return { ...prev, [id]: r }
    })
    setSaveState('dirty')
  }, [])

  const addComparison = useCallback((cmp: Comparison) => {
    setComparisons((prev) => [...prev, cmp])
    setSaveState('dirty')
  }, [])

  const undoComparison = useCallback(() => {
    if (comparisonsRef.current.length === 0) return
    setComparisons((prev) => prev.slice(0, -1))
    setSaveState('dirty')
  }, [])

  const switchMode = useCallback((m: Mode) => {
    if (modeRef.current === m) return
    setMode(m)
    setSaveState('dirty') // the chosen mode persists in the ratings doc
  }, [])

  // Debounced autosave whenever there are unsaved edits.
  useEffect(() => {
    if (saveState !== 'dirty') return
    const t = setTimeout(async () => {
      setSaveState('saving')
      try {
        await putRatings(roundName, {
          mode: modeRef.current,
          ratings: ratingsRef.current,
          comparisons: comparisonsRef.current,
          ranking:
            modeRef.current === 'pairwise'
              ? rankClips(clipsRef.current, comparisonsRef.current)
              : [],
        })
        // An edit made while the PUT was in flight re-marks state as dirty;
        // only advance to clean if nothing intervened.
        setSaveState((s) => (s === 'saving' ? 'clean' : s))
        onSaved()
      } catch {
        setSaveState((s) => (s === 'saving' ? 'error' : s))
      }
    }, SAVE_DEBOUNCE_MS)
    return () => clearTimeout(t)
  }, [saveState, ratings, comparisons, mode, roundName, onSaved])

  const registerAudio = useCallback((id: string, el: HTMLAudioElement | null) => {
    if (el) audioRefs.current.set(id, el)
    else audioRefs.current.delete(id)
  }, [])

  // Playing one card pauses all others.
  const handlePlay = useCallback((index: number) => {
    setCurrent(index)
    const playing = clipsRef.current[index]?.id
    for (const [id, el] of audioRefs.current) {
      if (id !== playing && !el.paused) el.pause()
    }
  }, [])

  const focusClip = useCallback((index: number, autoplay: boolean) => {
    const clip = clipsRef.current[index]
    if (!clip) return
    setCurrent(index)
    document
      .getElementById(`clip-${clip.id}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    if (autoplay) void audioRefs.current.get(clip.id)?.play()
  }, [])

  // Tiers-mode keyboard: space play/pause, 1/2/3 tier, a artifact, j/k
  // next/prev. Compare mode installs its own handler in CompareView.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (modeRef.current !== 'tiers') return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const target = e.target as HTMLElement
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)
        return
      if (target.tagName === 'BUTTON' || target.tagName === 'AUDIO') target.blur()

      const clip = clipsRef.current[currentRef.current]
      if (!clip) return
      const audio = audioRefs.current.get(clip.id)

      if (e.key === ' ') {
        e.preventDefault()
        if (audio) void (audio.paused ? audio.play() : audio.pause())
      } else if (e.key === '1') rate(clip.id, 'top')
      else if (e.key === '2') rate(clip.id, 'keep')
      else if (e.key === '3') rate(clip.id, 'pass')
      else if (e.key === 'a') rate(clip.id, 'flag')
      else if (e.key === 'j' || e.key === 'k') {
        audio?.pause()
        const next = Math.min(
          Math.max(currentRef.current + (e.key === 'j' ? 1 : -1), 0),
          clipsRef.current.length - 1,
        )
        focusClip(next, true)
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [rate, focusClip])

  if (clips === null) return <div className="screen-msg">Loading round…</div>

  const rated = clips.filter((c) => ratings[c.id]?.tier).length
  const tops = clips.filter((c) => ratings[c.id]?.tier === 'top').length

  return (
    <div className="round-screen">
      <header className="round-header">
        <div className="round-title-row">
          <h1>{roundName}</h1>
          <div className="mode-toggle" role="group" aria-label="Rating mode">
            <button
              type="button"
              className={mode === 'tiers' ? 'selected' : ''}
              aria-pressed={mode === 'tiers'}
              onClick={(e) => {
                switchMode('tiers')
                e.currentTarget.blur()
              }}
            >
              <IconRows /> Tiers
            </button>
            <button
              type="button"
              className={mode === 'pairwise' ? 'selected' : ''}
              aria-pressed={mode === 'pairwise'}
              onClick={(e) => {
                switchMode('pairwise')
                e.currentTarget.blur()
              }}
            >
              <IconScale /> Compare
            </button>
          </div>
        </div>
        <p className="lead">
          Grade on a curve, gut speed — the question is which of these are{' '}
          <em>most you</em>, not whether they are releasable.{' '}
          {mode === 'tiers' ? (
            <>
              <strong>Top pick</strong> = your best 2–4 of this hand (these become
              training data). <strong>Keep</strong> = plausible, right direction.{' '}
              <strong>Pass</strong> = not you.
            </>
          ) : (
            <>
              Forced choice: of each pair, pick whichever is <em>more</em> you —
              even when both are far off, the winner still carries signal.
            </>
          )}{' '}
          <strong>Artifacts</strong> is an independent flag for AI garble and codec
          junk — flagged clips never enter training, no matter how stylish. Clip
          names are deliberately meaningless.
        </p>
        <p className="lead keys">
          {mode === 'tiers' ? (
            <>
              Keyboard: <kbd>space</kbd> play/pause, <kbd>1</kbd> top pick,{' '}
              <kbd>2</kbd> keep, <kbd>3</kbd> pass, <kbd>a</kbd> artifact flag,{' '}
              <kbd>j</kbd>/<kbd>k</kbd> next/previous.
            </>
          ) : (
            <>
              Keyboard: <kbd>left</kbd>/<kbd>right</kbd> arrow picks the winner,{' '}
              <kbd>space</kbd> toggles play between A and B, <kbd>u</kbd> undoes
              the last comparison.
            </>
          )}
        </p>
      </header>

      {mode === 'tiers' ? (
        <div className="clips">
          {clips.map((clip, i) => (
            <ClipCard
              key={clip.id}
              clip={clip}
              index={i}
              total={clips.length}
              rating={ratings[clip.id] ?? {}}
              isCurrent={i === current}
              onRate={rate}
              onPlay={handlePlay}
              registerAudio={registerAudio}
            />
          ))}
        </div>
      ) : (
        <CompareView
          clips={clips}
          ratings={ratings}
          comparisons={comparisons}
          onCompare={addComparison}
          onUndo={undoComparison}
          onToggleArtifact={(id) => rate(id, 'flag')}
        />
      )}

      <footer className="progress-bar">
        <span className="progress-text">
          {mode === 'tiers' ? (
            <>
              {rated}/{clips.length} rated ({tops} top pick{tops === 1 ? '' : 's'})
            </>
          ) : (
            <>
              {comparisons.length}/~{targetComparisons(clips.length)} comparisons
            </>
          )}
        </span>
        <span className={`save-state save-${saveState}`}>
          {saveState === 'clean' && (
            <>
              <IconCircleCheck /> Saved
            </>
          )}
          {saveState === 'dirty' && (
            <>
              <IconDot /> Unsaved changes
            </>
          )}
          {saveState === 'saving' && (
            <>
              <IconLoader /> Saving…
            </>
          )}
          {saveState === 'error' && (
            <>
              <IconAlert /> Save failed — will retry on next change
            </>
          )}
        </span>
      </footer>
    </div>
  )
}
