import { useCallback, useEffect, useState } from 'react'
import { fetchRounds } from './api'
import { IconCircleCheck, IconMusic } from './icons'
import { RoundScreen } from './RoundScreen'
import type { RoundSummary } from './types'

function App() {
  const [rounds, setRounds] = useState<RoundSummary[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refreshRounds = useCallback(() => {
    fetchRounds()
      .then(setRounds)
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(refreshRounds, [refreshRounds])

  // Auto-select the first round once the list arrives.
  useEffect(() => {
    if (selected === null && rounds && rounds.length > 0) setSelected(rounds[0].name)
  }, [rounds, selected])

  return (
    <div className="app">
      <nav className="rail">
        <div className="rail-title">
          <IconMusic width={16} height={16} />
          Rounds
        </div>
        {error && <div className="rail-msg">API unreachable: {error}</div>}
        {rounds === null && !error && <div className="rail-msg">Loading…</div>}
        {rounds?.length === 0 && (
          <div className="rail-msg">
            No rounds yet. Deal one into app/data/rounds/ with deal_round.py.
          </div>
        )}
        {rounds?.map((r) => (
          <button
            type="button"
            key={r.name}
            className={`rail-item${r.name === selected ? ' selected' : ''}`}
            onClick={(e) => {
              setSelected(r.name)
              e.currentTarget.blur()
            }}
          >
            <span className="rail-name">{r.name}</span>
            <span className="rail-meta">
              {r.clipCount} clips
              {r.rated && <IconCircleCheck width={13} height={13} />}
            </span>
          </button>
        ))}
      </nav>
      <main>
        {selected ? (
          <RoundScreen key={selected} roundName={selected} onSaved={refreshRounds} />
        ) : (
          <div className="screen-msg">Select a round to start rating.</div>
        )}
      </main>
    </div>
  )
}

export default App
