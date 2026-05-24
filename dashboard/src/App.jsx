import { useRef, useState, useEffect } from 'react'
import { motion, useInView } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell, Legend,
} from 'recharts'
import vizData from './data/viz_data.json'
import ParticleBackground from './components/ParticleBackground'

const C = {
  geno: '#00D4AA',
  pheno: '#FF6B6B',
  accent: '#FF6A1A',
  gold: '#FFD700',
  pos: '#ef4444',
  neg: '#4ade80',
  muted: '#6b7280',
  grid: 'rgba(255,255,255,0.06)',
  axis: '#9ca3af',
  cardBg: 'rgba(255,255,255,0.04)',
  cardBorder: 'rgba(255,255,255,0.08)',
}

const STAGE_PALETTE = { adult_aging: '#FF6A1A', developmental: '#6366f1', postnatal: '#06b6d4', unspecified: '#6b7280' }

function Section({ children, className = '' }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })
  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 36 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      className={`max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 ${className}`}
    >
      {children}
    </motion.section>
  )
}

function Heading({ children, sub }) {
  return (
    <div className="mb-8">
      <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">{children}</h2>
      {sub && <p className="mt-2 text-gray-400 text-sm leading-relaxed max-w-3xl">{sub}</p>}
    </div>
  )
}

function Card({ children, className = '', glow }) {
  const border = glow === 'green' ? 'border-emerald-500/20' : glow === 'red' ? 'border-red-500/20' : glow === 'gold' ? 'border-yellow-500/20' : 'border-white/[0.08]'
  const shadow = glow === 'green' ? 'shadow-[0_0_30px_rgba(0,212,170,0.08)]' : glow === 'red' ? 'shadow-[0_0_30px_rgba(255,68,68,0.08)]' : ''
  return (
    <div className={`rounded-2xl border backdrop-blur-md p-5 sm:p-6 ${border} ${shadow} ${className}`} style={{ background: C.cardBg }}>
      {children}
    </div>
  )
}

function CountUp({ end, decimals = 0, prefix = '', suffix = '' }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (!inView) return
    const dur = 1600
    const start = performance.now()
    function tick(now) {
      const t = Math.min((now - start) / dur, 1)
      const ease = 1 - Math.pow(1 - t, 3)
      setVal(ease * end)
      if (t < 1) requestAnimationFrame(tick)
      else setVal(end)
    }
    requestAnimationFrame(tick)
  }, [inView, end])
  return <span ref={ref}>{prefix}{val.toFixed(decimals)}{suffix}</span>
}

function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-gray-900/95 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-300 font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.fill || p.color || p.stroke }}>
          {p.name}: {typeof p.value === 'number' && p.value <= 1 && p.name !== 'count' ? `${(p.value * 100).toFixed(1)}%` : p.value?.toLocaleString()}
        </p>
      ))}
    </div>
  )
}

function ConfusionMatrix({ confusion, label, color }) {
  const { tp, fp, fn, tn } = confusion
  return (
    <div>
      <h4 className="text-xs font-semibold mb-2 tracking-wide uppercase" style={{ color }}>{label}</h4>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center text-xs font-mono">
        <div />
        <div className="text-gray-500 py-1 text-[10px]">Pred +</div>
        <div className="text-gray-500 py-1 text-[10px]">Pred −</div>
        <div className="text-gray-500 py-1 pr-2 text-right text-[10px]">Act +</div>
        <div className="rounded bg-emerald-900/40 border border-emerald-500/25 py-2.5 text-emerald-400 font-bold">{tp}</div>
        <div className="rounded bg-red-900/30 border border-red-500/25 py-2.5 text-red-400 font-bold">{fn}</div>
        <div className="text-gray-500 py-1 pr-2 text-right text-[10px]">Act −</div>
        <div className="rounded bg-red-900/30 border border-red-500/25 py-2.5 text-red-400 font-bold">{fp}</div>
        <div className="rounded bg-emerald-900/40 border border-emerald-500/25 py-2.5 text-emerald-400 font-bold">{tn}</div>
      </div>
    </div>
  )
}

function MetricRow({ label, geno, pheno, fmt }) {
  const f = fmt || (v => (v * 100).toFixed(1) + '%')
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/5 text-sm">
      <span className="text-gray-400 w-32">{label}</span>
      <span className="font-mono font-semibold" style={{ color: C.geno }}>{f(geno)}</span>
      <span className="text-gray-600 mx-2">→</span>
      <span className="font-mono font-semibold" style={{ color: C.pheno }}>{f(pheno)}</span>
      <span className={`text-xs ml-3 font-mono ${geno > pheno ? 'text-red-400' : geno < pheno ? 'text-emerald-400' : 'text-gray-500'}`}>
        {geno !== pheno ? (geno > pheno ? `−${f(geno - pheno)}` : `+${f(pheno - geno)}`) : '—'}
      </span>
    </div>
  )
}

/* ── HERO ─────────────────────────────────────────────── */

function Hero() {
  const stats = [
    { v: vizData.dataset.total_genotypes, label: 'Genotypes Analyzed', color: C.accent },
    { v: vizData.eval.n_prompts, label: 'Benchmark Prompts', color: C.geno },
    { v: vizData.eval.delta_recall, label: 'Δ recall', color: C.gold, dec: 3, prefix: '+' },
  ]
  return (
    <header className="relative min-h-[85vh] flex flex-col items-center justify-center text-center overflow-hidden">
      <ParticleBackground />
      <div className="relative z-10 px-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
          <p className="text-xs tracking-[0.3em] uppercase text-gray-500 mb-4">Caltech Longevity Hackathon 2026 — Track 01</p>
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold leading-[1.05] tracking-tight">
            Mouse Longevity<br />
            <span className="bg-gradient-to-r from-[#00D4AA] via-[#00D4AA] to-[#FFD700] bg-clip-text text-transparent">Benchmark</span>
          </h1>
          <p className="mt-5 text-gray-400 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
            Can an aging-biology LLM predict mouse survival from genotype&nbsp;+&nbsp;phenotype?
            We built a 240-prompt ablation benchmark to find out — and measured how much the model
            leans on <span style={{ color: C.geno }}>memorized gene facts</span> vs.{' '}
            <span style={{ color: C.pheno }}>genuine reasoning</span>.
          </p>
        </motion.div>
        <motion.div
          className="mt-10 flex flex-wrap justify-center gap-4"
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.6 }}
        >
          {stats.map((s, i) => (
            <div key={i} className="rounded-xl border border-white/[0.08] backdrop-blur-md px-6 py-4 text-center" style={{ background: C.cardBg }}>
              <div className="text-2xl sm:text-3xl font-bold font-mono" style={{ color: s.color }}>
                <CountUp end={s.v} decimals={s.dec || 0} prefix={s.prefix || ''} />
              </div>
              <div className="text-[11px] text-gray-500 mt-1 tracking-wide uppercase">{s.label}</div>
            </div>
          ))}
        </motion.div>
        <motion.div
          className="mt-8 flex gap-3 justify-center"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="w-3 h-3 rounded-sm" style={{ background: C.geno }} /> Gene Shown (geno+pheno)
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="w-3 h-3 rounded-sm" style={{ background: C.pheno }} /> Gene Hidden (pheno only)
          </div>
        </motion.div>
      </div>
      <div className="absolute bottom-8 animate-bounce text-gray-600">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M10 14l-5-5h10l-5 5z" /></svg>
      </div>
    </header>
  )
}

/* ── PANEL 1 — Dataset Composition ────────────────────── */

function DatasetComposition() {
  const { mortality_category, death_by_stage, primary_system_top } = vizData.dataset

  const mortalityData = Object.entries(mortality_category)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))

  const stageData = Object.entries(death_by_stage)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name: name.replace('_', ' '), value }))

  const systemData = Object.entries(primary_system_top)
    .sort((a, b) => a[1] - b[1])
    .map(([name, value]) => ({ name, value }))

  return (
    <Section>
      <Heading
        sub="Developmental lethality dominates the dataset — only 3,640 of 18,465 mortality genotypes involve adult/aging, the longevity-relevant cases."
      >
        Dataset: <span style={{ color: C.accent }}>{vizData.dataset.total_genotypes.toLocaleString()}</span> Genotypes from MGI
      </Heading>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Mortality Category</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={mortalityData} margin={{ bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: C.axis, fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="value" name="count" radius={[3, 3, 0, 0]}>
                {mortalityData.map((e, i) => (
                  <Cell key={i} fill={e.name === 'Death' ? C.pos : e.name === 'None' ? C.neg : e.name === 'Reversed' ? C.gold : C.muted} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Mortality by Developmental Stage</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={stageData}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: C.axis, fontSize: 11 }} />
              <YAxis tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="value" name="count" radius={[3, 3, 0, 0]}>
                {stageData.map((e, i) => {
                  const key = e.name.replace(' ', '_')
                  return <Cell key={i} fill={STAGE_PALETTE[key] || C.muted} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card className="mt-6">
        <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Primary Affected System</h3>
        <ResponsiveContainer width="100%" height={440}>
          <BarChart layout="vertical" data={systemData} margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.grid} horizontal={false} />
            <XAxis type="number" tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v} />
            <YAxis dataKey="name" type="category" width={175} tick={{ fill: C.axis, fontSize: 11 }} />
            <Tooltip content={<Tip />} />
            <Bar dataKey="value" name="count" fill="#6366f1" radius={[0, 3, 3, 0]} barSize={18} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </Section>
  )
}

/* ── PANEL 2 — Label Correction ───────────────────────── */

function LabelCorrection() {
  const lc = vizData.label_correction
  const items = [
    { n: lc.reversed_flipped_to_negative, label: 'life-extending genotypes corrected', sub: 'Were mislabeled "impairs survival"', color: C.gold },
    { n: lc.root_leak_profiles_cleaned, label: 'root-leak profiles cleaned', sub: 'MP root term contamination', color: C.accent },
    { n: lc.contradictory_excluded, label: 'contradictory excluded', sub: 'Conflicting mortality annotations', color: C.pos },
    { n: lc.total_excluded, label: 'total records excluded', sub: 'Ambiguous + contradictory + reproductive', color: C.pheno },
  ]
  return (
    <Section>
      <Heading sub="The original MGI build mislabeled any mortality-ontology term as 'impairs survival'. We audited every label to fix direction errors and exclude ambiguous records.">
        Label Correction
      </Heading>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {items.map((it, i) => (
          <Card key={i}>
            <div className="text-3xl font-bold font-mono" style={{ color: it.color }}>
              <CountUp end={it.n} />
            </div>
            <div className="text-sm font-medium text-gray-300 mt-1">{it.label}</div>
            <div className="text-xs text-gray-500 mt-1">{it.sub}</div>
          </Card>
        ))}
      </div>
      <div className="mt-4 rounded-xl border border-yellow-500/20 bg-yellow-500/[0.04] px-5 py-3 text-sm text-gray-400">
        <span className="font-semibold text-yellow-400">Key fix:</span> 407 genotypes annotated with longevity-extending MP terms
        (e.g. <span className="font-mono text-xs text-yellow-300">MP:0006080 increased lifespan</span>) were originally counted as "impairs survival" &mdash; flipped to negative.
      </div>
    </Section>
  )
}

/* ── PANEL 3 — Aging Direction Skew ───────────────────── */

function AgingDirection() {
  const ad = vizData.aging_direction
  const bars = [
    { name: 'Shortens lifespan', all: ad.shortens.all, usable: ad.shortens.usable, fill: C.pos },
    { name: 'Protective', all: ad.protective.all, usable: ad.protective.usable, fill: '#60a5fa' },
    { name: 'Extends lifespan', all: ad.extends.all, usable: ad.extends.usable, fill: C.geno },
  ]
  return (
    <Section>
      <Heading sub="Genuine life-extending positives are extraordinarily rare — a key design constraint for building a balanced benchmark.">
        Aging-Direction Skew
      </Heading>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-8 items-start">
        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Usable Genotypes by Aging Direction</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={bars} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} horizontal={false} />
              <XAxis type="number" tick={{ fill: C.axis, fontSize: 10 }} />
              <YAxis dataKey="name" type="category" width={130} tick={{ fill: C.axis, fontSize: 12 }} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="usable" name="count" radius={[0, 4, 4, 0]} barSize={28}>
                {bars.map((b, i) => <Cell key={i} fill={b.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className="lg:w-64" glow="gold">
          <div className="text-center">
            <div className="text-6xl font-bold font-mono" style={{ color: C.gold }}>
              <CountUp end={51} />
            </div>
            <div className="text-sm text-gray-400 mt-2">usable life-extending genotypes</div>
            <div className="text-xs text-gray-600 mt-1">out of {ad.shortens.usable.toLocaleString()} that shorten lifespan</div>
            <div className="mt-4 text-5xl font-bold font-mono text-gray-600">
              66:1
            </div>
            <div className="text-xs text-gray-500 mt-1">shortens-to-extends ratio</div>
          </div>
        </Card>
      </div>
    </Section>
  )
}

/* ── PANEL 4 — Benchmark Composition ──────────────────── */

function BenchmarkComposition() {
  const bc = vizData.benchmark_composition
  const controlled = bc.controlled.positive_stage
  const random = bc.random.positive_stage

  const chartData = [
    {
      name: 'Controlled',
      adult_aging: controlled.adult_aging,
      developmental: controlled.developmental,
      postnatal: 0,
      total: bc.controlled.pos,
    },
    {
      name: 'Random',
      adult_aging: random.adult_aging,
      developmental: random.developmental,
      postnatal: random.postnatal || 0,
      total: bc.random.pos,
    },
  ]

  return (
    <Section>
      <Heading sub="Random sampling pulls 72% developmental cases (easy for the model). Our controlled curation forces adult/aging — the longevity-relevant, harder cases.">
        Benchmark Composition: <span style={{ color: C.geno }}>Controlled</span> vs <span style={{ color: C.pheno }}>Random</span>
      </Heading>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Positive Cases by Stage (n=60 each)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: C.axis, fontSize: 13 }} />
              <YAxis tick={{ fill: C.axis, fontSize: 10 }} />
              <Tooltip content={<Tip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.axis }} />
              <Bar dataKey="adult_aging" name="Adult / Aging" stackId="a" fill={C.accent} radius={[0, 0, 0, 0]} />
              <Bar dataKey="developmental" name="Developmental" stackId="a" fill="#6366f1" radius={[0, 0, 0, 0]} />
              <Bar dataKey="postnatal" name="Postnatal" stackId="a" fill="#06b6d4" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <div className="grid grid-rows-2 gap-4">
          <Card glow="green">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] tracking-[0.2em] uppercase font-semibold px-2 py-0.5 rounded-full border border-emerald-500/30 text-emerald-400">Controlled</span>
                <p className="text-gray-400 text-sm mt-3">
                  <span className="text-2xl font-bold font-mono text-white">80%</span> adult / aging
                </p>
                <p className="text-xs text-gray-500 mt-1">{bc.controlled.n} prompts &middot; {bc.controlled.n_systems} body systems &middot; balanced 60/60</p>
              </div>
              <div className="text-4xl font-bold font-mono" style={{ color: C.geno }}>48</div>
            </div>
            <p className="text-xs text-gray-500 mt-3">Forces the model to reason about adult-onset phenotypes — the cases that matter for longevity.</p>
          </Card>

          <Card glow="red">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] tracking-[0.2em] uppercase font-semibold px-2 py-0.5 rounded-full border border-red-500/30 text-red-400">Random</span>
                <p className="text-gray-400 text-sm mt-3">
                  <span className="text-2xl font-bold font-mono text-white">72%</span> developmental
                </p>
                <p className="text-xs text-gray-500 mt-1">{bc.random.n} prompts &middot; {bc.random.n_systems} body systems &middot; balanced 60/60</p>
              </div>
              <div className="text-4xl font-bold font-mono" style={{ color: C.pheno }}>43</div>
            </div>
            <p className="text-xs text-gray-500 mt-3">Random sampling yields mostly embryonic lethals — easy to flag, but not longevity-informative.</p>
          </Card>
        </div>
      </div>
    </Section>
  )
}

/* ── PANEL 5 — Eval Headline: Δ_recall (hero) ────────── */

function EvalHeadline() {
  const ev = vizData.eval
  const gp = ev.by_condition.geno_pheno
  const po = ev.by_condition.pheno_only

  const metricsData = [
    { name: 'Accuracy', geno: gp.accuracy, pheno: po.accuracy },
    { name: 'Bal. Acc.', geno: gp.balanced_accuracy, pheno: po.balanced_accuracy },
    { name: 'MCC', geno: gp.mcc, pheno: po.mcc },
    { name: 'Sensitivity', geno: gp.sensitivity, pheno: po.sensitivity },
    { name: 'Specificity', geno: gp.specificity, pheno: po.specificity },
  ]

  return (
    <Section>
      <div className="text-center mb-10">
        <p className="text-xs tracking-[0.3em] uppercase text-gray-500 mb-2">The Ablation Result</p>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight">
          Δ<sub>recall</sub> ={' '}
          <span className="bg-gradient-to-r from-[#FFD700] to-[#FF6A1A] bg-clip-text text-transparent">+0.125</span>
        </h2>
        <p className="mt-3 text-gray-400 text-sm max-w-2xl mx-auto leading-relaxed">
          Hiding the gene name drops balanced accuracy <span style={{ color: C.geno }}>0.81</span> →{' '}
          <span style={{ color: C.pheno }}>0.68</span>. The drop is mostly in specificity
          (<span style={{ color: C.geno }}>0.77</span> → <span style={{ color: C.pheno }}>0.60</span>)
          — without the gene, the model gets trigger-happy and over-predicts "impairs survival."
          <br />Gene recall mostly helps the model <strong>rule out</strong> death.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Gene Shown vs Gene Hidden</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={metricsData} margin={{ bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: C.axis, fontSize: 12 }} />
              <YAxis tick={{ fill: C.axis, fontSize: 10 }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip content={<Tip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="geno" name="Gene Shown" fill={C.geno} radius={[3, 3, 0, 0]} barSize={32} />
              <Bar dataKey="pheno" name="Gene Hidden" fill={C.pheno} radius={[3, 3, 0, 0]} barSize={32} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <div className="flex flex-col gap-4 lg:w-72">
          <Card>
            <ConfusionMatrix confusion={gp.confusion} label="Gene Shown (geno+pheno)" color={C.geno} />
          </Card>
          <Card>
            <ConfusionMatrix confusion={po.confusion} label="Gene Hidden (pheno only)" color={C.pheno} />
          </Card>
        </div>
      </div>

      <Card className="mt-6">
        <h3 className="text-sm font-semibold text-gray-300 mb-3 tracking-wide uppercase">Metric Breakdown</h3>
        <div className="grid grid-cols-[auto_1fr] gap-x-6 text-xs text-gray-500 mb-2 px-1">
          <span />
          <div className="flex justify-between">
            <span style={{ color: C.geno }}>Gene Shown</span>
            <span />
            <span style={{ color: C.pheno }}>Gene Hidden</span>
            <span>Drop</span>
          </div>
        </div>
        <MetricRow label="Accuracy" geno={gp.accuracy} pheno={po.accuracy} />
        <MetricRow label="Balanced Acc." geno={gp.balanced_accuracy} pheno={po.balanced_accuracy} />
        <MetricRow label="MCC" geno={gp.mcc} pheno={po.mcc} fmt={v => v.toFixed(2)} />
        <MetricRow label="Sensitivity" geno={gp.sensitivity} pheno={po.sensitivity} />
        <MetricRow label="Specificity" geno={gp.specificity} pheno={po.specificity} />
        <div className="mt-3 text-xs text-gray-500 italic">
          Positive = "impairs survival"; Negative = no mortality phenotype. Sensitivity = correctly flagged lethal genotypes; Specificity = correctly cleared survivors.
        </div>
      </Card>
    </Section>
  )
}

/* ── PANEL 6 — Eval by Axis ───────────────────────────── */

function EvalByAxis() {
  const { by_stage, by_system } = vizData.eval

  const stageData = Object.entries(by_stage).map(([name, v]) => ({
    name: name.replace('_', ' '),
    geno: v.geno_pheno,
    pheno: v.pheno_only,
    delta: v.geno_pheno - v.pheno_only,
  }))

  const systemData = Object.entries(by_system)
    .map(([name, v]) => ({
      name,
      geno: v.geno_pheno,
      pheno: v.pheno_only,
      delta: v.geno_pheno - v.pheno_only,
    }))
    .sort((a, b) => b.delta - a.delta)

  return (
    <Section>
      <Heading sub="Where does the model lean hardest on memorized gene identity? Cardiovascular is the most recall-dependent system (0.70 → 0.40, Δ = 0.30).">
        Evaluation by Axis
      </Heading>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">By Developmental Stage</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={stageData}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
              <XAxis dataKey="name" tick={{ fill: C.axis, fontSize: 12 }} />
              <YAxis tick={{ fill: C.axis, fontSize: 10 }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip content={<Tip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="geno" name="Gene Shown" fill={C.geno} radius={[3, 3, 0, 0]} barSize={36} />
              <Bar dataKey="pheno" name="Gene Hidden" fill={C.pheno} radius={[3, 3, 0, 0]} barSize={36} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">Δ by Stage</h3>
          <div className="flex flex-col justify-center h-[180px] gap-6 px-4">
            {stageData.map((s, i) => (
              <div key={i}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-300 capitalize">{s.name}</span>
                  <span className="font-mono text-yellow-400">Δ = {(s.delta * 100).toFixed(0)}%</span>
                </div>
                <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: `linear-gradient(90deg, ${C.geno}, ${C.gold})` }}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${s.delta * 100 * 10}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1, ease: 'easeOut' }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="mt-6">
        <h3 className="text-sm font-semibold text-gray-300 mb-4 tracking-wide uppercase">By Body System (sorted by gene-recall dependency)</h3>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={systemData} layout="vertical" margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.grid} horizontal={false} />
            <XAxis type="number" tick={{ fill: C.axis, fontSize: 10 }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <YAxis dataKey="name" type="category" width={180} tick={{ fill: C.axis, fontSize: 11 }} />
            <Tooltip content={<Tip />} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="geno" name="Gene Shown" fill={C.geno} radius={[0, 3, 3, 0]} barSize={14} />
            <Bar dataKey="pheno" name="Gene Hidden" fill={C.pheno} radius={[0, 3, 3, 0]} barSize={14} />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-3 flex flex-wrap gap-3">
          {systemData.filter(s => s.delta >= 0.2).map((s, i) => (
            <span key={i} className="text-xs px-3 py-1 rounded-full border border-yellow-500/30 text-yellow-400 bg-yellow-500/[0.05]">
              {s.name}: Δ = {(s.delta * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      </Card>
    </Section>
  )
}

/* ── FOOTER ───────────────────────────────────────────── */

function Footer() {
  return (
    <footer className="border-t border-white/[0.06] mt-16">
      <div className="max-w-7xl mx-auto px-6 py-12 text-center">
        <p className="text-sm text-gray-400">Built at <span className="text-white font-medium">Caltech Longevity Hackathon 2026</span></p>
        <p className="text-xs text-gray-600 mt-2">Data: Mouse Genome Informatics (MGI) &middot; International Mouse Phenotyping Consortium (IMPC)</p>
        <p className="text-xs text-gray-600 mt-1">Model: Longevity-LLM &middot; Benchmark: LongevityBench format</p>
        <div className="mt-4">
          <span className="text-[10px] tracking-[0.2em] uppercase font-semibold px-3 py-1 rounded-full border border-emerald-500/30 text-emerald-400">
            Track 01 — LongevityLLM Benchmarking
          </span>
        </div>
      </div>
    </footer>
  )
}

/* ── APP ──────────────────────────────────────────────── */

export default function App() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F2EEE6] font-sans">
      <Hero />
      <DatasetComposition />
      <LabelCorrection />
      <AgingDirection />
      <BenchmarkComposition />
      <EvalHeadline />
      <EvalByAxis />
      <Footer />
    </div>
  )
}
