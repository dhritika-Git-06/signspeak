import * as Slider from "@radix-ui/react-slider"
import * as Switch from "@radix-ui/react-switch"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import {
  forwardRef,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type LabelHTMLAttributes,
  type ReactNode,
} from "react"
import { cn } from "@/lib/cn"

const buttonStyles = cva(
  "inline-flex items-center justify-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-[var(--accent)] text-[var(--accent-ink)] hover:brightness-110",
        quiet: "border border-[var(--line)] bg-[var(--bg-elev)] hover:bg-[var(--bg)]",
        ghost: "hover:bg-[var(--bg)]",
        danger: "border border-[var(--line)] text-[var(--accent)] hover:bg-[var(--bg)]",
      },
      size: {
        md: "h-10",
        sm: "h-8 px-3 text-xs",
        lg: "h-12 px-5",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
)

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonStyles> {
  asChild?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant, size, asChild, ...props },
  ref,
) {
  const Comp = asChild ? Slot : "button"
  return <Comp ref={ref} className={cn(buttonStyles({ variant, size }), className)} {...props} />
})

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...props },
  ref,
) {
  return (
    <input
      ref={ref}
      className={cn(
        "h-11 w-full rounded-xl border border-[var(--line)] bg-[var(--bg)] px-3 text-[var(--ink)] outline-none",
        className,
      )}
      {...props}
    />
  )
})

export function Label({ className, ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("text-sm font-medium", className)} {...props} />
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "warn" }) {
  const tones = {
    neutral: "border-[var(--line)] text-[var(--muted)]",
    good: "border-transparent bg-[color-mix(in_srgb,var(--good)_18%,transparent)] text-[var(--good)]",
    warn: "border-transparent bg-[color-mix(in_srgb,var(--warn)_18%,transparent)] text-[var(--warn)]",
  }
  return (
    <span className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium", tones[tone])}>
      {children}
    </span>
  )
}

export function SwitchControl({
  checked,
  onCheckedChange,
  label,
  id,
}: {
  checked: boolean
  onCheckedChange: (value: boolean) => void
  label: string
  id: string
}) {
  return (
    <div className="flex items-center gap-2">
      <Switch.Root
        id={id}
        checked={checked}
        onCheckedChange={onCheckedChange}
        className="h-6 w-11 rounded-full bg-[var(--line)] data-[state=checked]:bg-[var(--accent)]"
      >
        <Switch.Thumb className="block h-5 w-5 translate-x-0.5 rounded-full bg-white transition data-[state=checked]:translate-x-[22px]" />
      </Switch.Root>
      <Label htmlFor={id}>{label}</Label>
    </div>
  )
}

export function SliderControl({
  value,
  onChange,
  min,
  max,
  step,
  label,
  id,
}: {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  label: string
  id: string
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-sm">
        <Label htmlFor={id}>{label}</Label>
        <span className="tabular-nums text-[var(--muted)]">{Math.round(value * 100)}%</span>
      </div>
      <Slider.Root
        id={id}
        className="relative flex h-5 items-center"
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(next) => onChange(next[0] ?? value)}
      >
        <Slider.Track className="relative h-1.5 grow rounded-full bg-[var(--line)]">
          <Slider.Range className="absolute h-full rounded-full bg-[var(--accent)]" />
        </Slider.Track>
        <Slider.Thumb className="block h-4 w-4 rounded-full border border-[var(--accent)] bg-[var(--bg-elev)]" />
      </Slider.Root>
    </div>
  )
}
