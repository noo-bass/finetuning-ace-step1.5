import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { IconFlag, IconPause, IconPlay, IconStar, IconUndo } from './icons'
import { nextPair, rankClips, targetComparisons } from './pairing'
import type { Clip, Comparison, Ratings } from './types'

const TOP_KEEPERS = 4

interface Props {
  clips: Clip[]
  ratings: Ratings
  comparisons: Comparison[]
  onCompare: (cmp: Comparison) => void
  onUndo: () => void
  onToggleArtifact: (id: string) => void
}

export function CompareView({
  clips,
  ratings,
  comparisons,
  onCompare,
  onUndo,
  onToggleArtifact,
}: Props) {
  const [finished, setFinished] = useState(false)
  const [activeSide, setActiveSide] = useState<'a' | 'b'>('a')
  const [resultPlayingId, setResultPlayingId] = useState<string | null>(null)

  const audioA = useRef<HTMLAudioElement | null>(null)
  const audioB = useRef<HTMLAudioElement | null>(null)
  const resultAudio = useRef<HTMLAudioElement | null>(null)

  const clipById = useMemo(
    () => new Map(clips.map((c) => [c.id, c])),
    [clips],
  )
  const pair = useMemo(() => nextPair(clips, comparisons), [clips, comparisons])
  const ranking = useMemo(() => rankClips(clips, comparisons), [clips, comparisons])
  const target = targetComparisons(clips.length)
  const stable = comparisons.length >= target

  const pick = useCallback(
    (winner: 'a' | 'b') => {
      if (!pair) return
      audioA.current?.pause()
      audioB.current?.pause()
      onCompare({
        a: pair[0],
        b: pair[1],
        winner: winner === 'a' ? pair[0] : pair[1],
        ts: new Date().toISOString(),
      })
      setActiveSide('a')
    },
    [pair, onCompare],
  )

  // Keyboard: left/right pick winner, space toggles play A<->B, u undoes.
  useEffect(() => {
    if (finished) return
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const el = e.target as HTMLElement
      if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return
      if (el.tagName === 'BUTTON' || el.tagName === 'AUDIO') el.blur()

      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        pick('a')
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        pick('b')
      } else if (e.key === ' ') {
        e.preventDefault()
        const a = audioA.current
        const b = audioB.current
        if (a && !a.paused) {
          a.pause()
          void b?.play()
          setActiveSide('b')
        } else if (b && !b.paused) {
          b.pause()
          void a?.play()
          setActiveSide('a')
        } else {
          void (activeSide === 'a' ? a : b)?.play()
        }
      } else if (e.key === 'u') {
        onUndo()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [finished, pick, onUndo, activeSide])

  // Results-panel playback: one shared player, one row playing at a time.
  const toggleResultPlay = (clip: Clip) => {
    const el = resultAudio.current
    if (!el) return
    if (resultPlayingId === clip.id) {
      el.pause()
      setResultPlayingId(null)
      return
    }
    el.src = clip.url
    void el.play()
    setResultPlayingId(clip.id)
  }

  const rankedList = (compact: boolean) => (
    <ol className={`ranked-list${compact ? ' compact' : ''}`}>
      {ranking.map((id, i) => {
        const clip = clipById.get(id)
        if (!clip) return null
        const artifact = !!ratings[id]?.artifact
        return (
          <li key={id} className={i < TOP_KEEPERS ? 'keeper' : ''}>
            <span className="rank-n">{i + 1}</span>
            <button
              type="button"
              className="icon-btn"
              aria-label={resultPlayingId === id ? `Pause ${id}` : `Play ${id}`}
              onClick={(e) => {
                toggleResultPlay(clip)
                e.currentTarget.blur()
              }}
            >
              {resultPlayingId === id ? <IconPause /> : <IconPlay />}
            </button>
            <span className="clip-id">{id}</span>
            {i < TOP_KEEPERS && (
              <span className="keeper-badge">
                <IconStar /> keeper
              </span>
            )}
            {artifact && (
              <span className="artifact-badge">
                <IconFlag /> artifacts
              </span>
            )}
          </li>
        )
      })}
    </ol>
  )

  if (clips.length < 2)
    return <div className="screen-msg">Compare mode needs at least two clips.</div>

  // One shared hidden player for ranked-list playback, present in both the
  // results panel and the mid-compare stable banner.
  const sharedPlayer = (
    <audio
      ref={resultAudio}
      onEnded={() => setResultPlayingId(null)}
      onPause={() => setResultPlayingId(null)}
    />
  )

  if (finished) {
    return (
      <div className="compare-results">
        {sharedPlayer}
        <h2>Ranking after {comparisons.length} comparisons</h2>
        <p className="lead">
          Best-first by pairwise wins. The top {TOP_KEEPERS} are this round's
          keepers.
        </p>
        {rankedList(false)}
        <div className="compare-actions">
          <button type="button" className="tier-btn" onClick={() => setFinished(false)}>
            <IconUndo /> Keep comparing
          </button>
        </div>
      </div>
    )
  }

  const sideCard = (side: 'a' | 'b') => {
    if (!pair) return null
    const id = side === 'a' ? pair[0] : pair[1]
    const clip = clipById.get(id)
    if (!clip) return null
    const artifact = !!ratings[id]?.artifact
    const ref = side === 'a' ? audioA : audioB
    const other = side === 'a' ? audioB : audioA
    return (
      <div className={`compare-card${activeSide === side ? ' current' : ''}`}>
        <header>
          <span className="side-label">{side.toUpperCase()}</span>
          <span className="clip-id">{id}</span>
        </header>
        <audio
          controls
          preload="metadata"
          src={clip.url}
          ref={ref}
          onPlay={() => {
            other.current?.pause()
            setActiveSide(side)
          }}
        />
        <div className="controls">
          <button
            type="button"
            className="tier-btn pick-btn"
            onClick={(e) => {
              pick(side)
              e.currentTarget.blur()
            }}
          >
            <IconStar />
            This one ({side === 'a' ? 'left' : 'right'} arrow)
          </button>
          <button
            type="button"
            className={`tier-btn tier-flag${artifact ? ' active' : ''}`}
            aria-pressed={artifact}
            onClick={(e) => {
              onToggleArtifact(id)
              e.currentTarget.blur()
            }}
          >
            <IconFlag />
            Artifacts
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="compare-view">
      {sharedPlayer}
      <div className="compare-status">
        <span className="progress-text">
          Comparison {comparisons.length + 1} of ~{target}
        </span>
        <span className="compare-status-actions">
          <button
            type="button"
            className="tier-btn"
            disabled={comparisons.length === 0}
            onClick={(e) => {
              onUndo()
              e.currentTarget.blur()
            }}
          >
            <IconUndo /> Undo (u)
          </button>
          <button
            type="button"
            className="tier-btn"
            disabled={comparisons.length === 0}
            onClick={(e) => {
              setFinished(true)
              e.currentTarget.blur()
            }}
          >
            Show results
          </button>
        </span>
      </div>

      <div className="compare-pair">
        {sideCard('a')}
        {sideCard('b')}
      </div>

      {stable && (
        <div className="stable-banner">
          <p>
            <strong>Ranking stable</strong> — finish or keep comparing to refine
            the middle of the field.
          </p>
          {rankedList(true)}
          <div className="compare-actions">
            <button
              type="button"
              className="tier-btn finish-btn"
              onClick={() => setFinished(true)}
            >
              Finish
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
