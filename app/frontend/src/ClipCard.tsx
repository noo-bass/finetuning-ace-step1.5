import type { Clip, Rating, Tier } from './types'
import { IconCheck, IconFlag, IconStar, IconX } from './icons'

interface Props {
  clip: Clip
  index: number
  total: number
  rating: Rating
  isCurrent: boolean
  onRate: (id: string, action: Tier | 'flag') => void
  onPlay: (index: number) => void
  registerAudio: (id: string, el: HTMLAudioElement | null) => void
}

export function ClipCard({
  clip,
  index,
  total,
  rating,
  isCurrent,
  onRate,
  onPlay,
  registerAudio,
}: Props) {
  const tierBtn = (tier: Tier, label: string, icon: React.ReactNode) => (
    <button
      type="button"
      className={`tier-btn tier-${tier}${rating.tier === tier ? ' active' : ''}`}
      aria-pressed={rating.tier === tier}
      onClick={(e) => {
        onRate(clip.id, tier)
        e.currentTarget.blur()
      }}
    >
      {icon}
      {label}
    </button>
  )

  return (
    <article
      id={`clip-${clip.id}`}
      className={`clip${rating.tier ? ' done' : ''}${isCurrent ? ' current' : ''}`}
    >
      <header>
        <span className="clip-id">{clip.id}</span>
        <span className="clip-count">
          {index + 1}/{total}
        </span>
      </header>
      <audio
        controls
        preload="metadata"
        src={clip.url}
        ref={(el) => registerAudio(clip.id, el)}
        onPlay={() => onPlay(index)}
      />
      <div className="controls">
        {tierBtn('top', 'Top pick', <IconStar />)}
        {tierBtn('keep', 'Keep', <IconCheck />)}
        {tierBtn('pass', 'Pass', <IconX />)}
        <button
          type="button"
          className={`tier-btn tier-flag${rating.artifact ? ' active' : ''}`}
          aria-pressed={!!rating.artifact}
          onClick={(e) => {
            onRate(clip.id, 'flag')
            e.currentTarget.blur()
          }}
        >
          <IconFlag />
          Artifacts
        </button>
      </div>
    </article>
  )
}
