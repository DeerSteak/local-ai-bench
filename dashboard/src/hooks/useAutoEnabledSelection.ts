import { useCallback, useEffect, useRef, useState } from "react";

export function useAutoEnabledSelection(items: string[]) {
  const [enabled, setEnabled] = useState<Set<string>>(new Set());
  const seenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const additions = items.filter(item => !seenRef.current.has(item));
    if (!additions.length) return;
    setEnabled(previous => new Set([...previous, ...additions]));
    additions.forEach(item => seenRef.current.add(item));
  }, [items]);

  const toggle = useCallback((item: string) => {
    setEnabled(previous => {
      const next = new Set(previous);
      if (next.has(item)) next.delete(item); else next.add(item);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    seenRef.current = new Set();
    setEnabled(new Set());
  }, []);

  return { enabled, toggle, reset };
}
