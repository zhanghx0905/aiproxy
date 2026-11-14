// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { useForm } from 'react-hook-form'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Form } from '@/components/ui/form'
import type { ModelCreateForm } from '@/validation/model'
import { RerankWindowFields } from './RerankWindowFields'

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))

describe('rerank window configuration', () => {
    let container: HTMLDivElement
    let root: Root

    beforeEach(() => {
        Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
        container = document.createElement('div')
        document.body.appendChild(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        container.remove()
    })

    it('loads saved limits and lets the user disable windowing', async () => {
        let current: (() => ModelCreateForm) | undefined
        function Harness() {
            const form = useForm<ModelCreateForm>({ defaultValues: {
                model: 'bce', type: 10,
                plugin: { rerank_window: { enable: true, max_tokens: 512, overlap_tokens: 80, batch_size: 16 } },
            } })
            current = form.getValues
            return <Form {...form}><RerankWindowFields control={form.control} /></Form>
        }
        await act(async () => root.render(<Harness />))
        const inputs = Array.from(container.querySelectorAll<HTMLInputElement>('input[type="number"]'))
        expect(inputs.map(input => input.value)).toContain('512')
        expect(inputs.map(input => input.value)).toContain('16')
        const toggle = container.querySelector<HTMLButtonElement>('[role="switch"]')!
        expect(toggle.getAttribute('aria-checked')).toBe('true')
        await act(async () => toggle.click())
        expect(current?.().plugin?.rerank_window?.enable).toBe(false)
        expect(container.querySelector('input[type="number"]')).toBeNull()
    })
})
