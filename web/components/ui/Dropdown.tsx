"use client";

import { Check, ChevronDown } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type RefObject } from "react";

export interface DropdownOption {
  value: string;
  label: string;
}

export function useDismiss(open: boolean, onClose: () => void, ref: RefObject<HTMLElement>): void {
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) onClose();
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open, onClose, ref]);
}

interface DropdownProps {
  label: string;
  options: DropdownOption[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  disabledReason?: string;
}

export function Dropdown({ label, options, value, onChange, disabled, disabledReason }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const baseId = useId();
  const listId = `${baseId}-list`;

  const selectedIndex = Math.max(options.findIndex((o) => o.value === value), 0);
  const selected = options[selectedIndex];
  const close = useCallback(() => setOpen(false), []);
  useDismiss(open, close, rootRef);

  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  const openMenu = () => {
    setActive(selectedIndex);
    setOpen(true);
  };

  const commit = (index: number) => {
    setOpen(false);
    triggerRef.current?.focus();
    if (options[index] && options[index].value !== value) onChange(options[index].value);
  };

  const onTriggerKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (["ArrowDown", "ArrowUp"].includes(event.key)) {
      event.preventDefault();
      openMenu();
    }
  };

  const onListKeyDown = (event: KeyboardEvent<HTMLUListElement>) => {
    const last = options.length - 1;
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActive((i) => Math.min(i + 1, last));
        break;
      case "ArrowUp":
        event.preventDefault();
        setActive((i) => Math.max(i - 1, 0));
        break;
      case "Home":
        event.preventDefault();
        setActive(0);
        break;
      case "End":
        event.preventDefault();
        setActive(last);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        commit(active);
        break;
      case "Escape":
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  return (
    <div ref={rootRef} className="relative print:hidden" title={disabled ? disabledReason : undefined}>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        onClick={() => (open ? close() : openMenu())}
        onKeyDown={onTriggerKeyDown}
        className={`flex items-center gap-2 h-8 px-3 bg-surface border text-xs transition-colors ${
          disabled
            ? "border-border-formal text-text-muted opacity-60 cursor-not-allowed"
            : open
              ? "border-secondary-blue text-text-secondary"
              : "border-border-formal text-text-secondary hover:border-secondary-blue/50"
        } focus-visible:outline focus-visible:outline-2 focus-visible:outline-secondary-blue`}
      >
        <span className="text-text-muted">{label}:</span>
        <span className={`font-medium ${disabled ? "text-text-muted" : "text-text-primary"}`}>{selected?.label}</span>
        <ChevronDown className={`h-3 w-3 text-text-muted transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </button>

      {open && (
        <ul
          ref={listRef}
          id={listId}
          role="listbox"
          tabIndex={-1}
          aria-label={label}
          aria-activedescendant={`${baseId}-opt-${active}`}
          onKeyDown={onListKeyDown}
          className="absolute left-0 top-full z-30 mt-px min-w-full w-max bg-surface border border-primary-navy shadow-flat outline-none"
        >
          {options.map((option, i) => {
            const isSelected = option.value === value;
            return (
              <li
                key={option.value}
                id={`${baseId}-opt-${i}`}
                role="option"
                aria-selected={isSelected}
                onMouseEnter={() => setActive(i)}
                onClick={() => commit(i)}
                className={`flex items-center justify-between gap-6 px-3 py-1.5 text-xs cursor-pointer ${
                  i === active ? "bg-subtle" : ""
                } ${isSelected ? "font-semibold text-primary-navy" : "text-text-secondary"}`}
              >
                <span>{option.label}</span>
                {isSelected && <Check className="h-3 w-3 text-secondary-blue" aria-hidden />}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
