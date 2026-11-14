import { useWatch, type Control } from 'react-hook-form'
import { useTranslation } from 'react-i18next'
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import type { ModelCreateForm } from '@/validation/model'

const numberFields = [
    { name: 'max_tokens', fallback: 512, min: 16, max: 131072 },
    { name: 'overlap_tokens', fallback: 80, min: 0, max: 131071 },
    { name: 'special_tokens', fallback: 4, min: 0, max: 131071 },
    { name: 'safety_tokens', fallback: 8, min: 0, max: 131071 },
    { name: 'batch_size', fallback: 32, min: 1, max: 128 },
    { name: 'max_document_tokens', fallback: 32768, min: 1, max: 131072 },
    { name: 'max_windows', fallback: 2048, min: 1, max: 8192 },
    { name: 'timeout_seconds', fallback: 180, min: 1, max: 600 },
] as const

/** Model-scoped text rerank windowing, shared by create and edit forms. */
export function RerankWindowFields({ control }: { control: Control<ModelCreateForm> }) {
    const { t } = useTranslation()
    const enabled = useWatch({ control, name: 'plugin.rerank_window.enable' })

    return (
        <div className="space-y-4 border-t py-4">
            <FormField control={control} name="plugin.rerank_window.enable" render={({ field }) => (
                <FormItem className="flex items-start justify-between gap-4">
                    <div>
                        <FormLabel>{t('model.dialog.rerankWindow.title')}</FormLabel>
                        <FormDescription>{t('model.dialog.rerankWindow.description')}</FormDescription>
                    </div>
                    <FormControl>
                        <Switch aria-label={t('model.dialog.rerankWindow.title')} checked={field.value ?? false} onCheckedChange={field.onChange} />
                    </FormControl>
                </FormItem>
            )} />
            {enabled && (
                <div className="grid gap-4 sm:grid-cols-2">
                    {numberFields.map(({ name, fallback, min, max }) => (
                        <FormField key={name} control={control} name={`plugin.rerank_window.${name}`} render={({ field }) => (
                            <FormItem>
                                <FormLabel>{t(`model.dialog.rerankWindow.${name}`)}</FormLabel>
                                <FormControl>
                                    <Input {...field} value={field.value ?? ''} type="number" step={1} min={min} max={max} placeholder={String(fallback)}
                                        onChange={event => field.onChange(event.target.value === '' ? undefined : Number(event.target.value))} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )} />
                    ))}
                    <FormField control={control} name="plugin.rerank_window.tokenizer_url" render={({ field }) => (
                        <FormItem className="sm:col-span-2">
                            <FormLabel>{t('model.dialog.rerankWindow.tokenizer_url')}</FormLabel>
                            <FormControl><Input {...field} value={field.value ?? ''} placeholder="http://tei:80/tokenize" /></FormControl>
                            <FormDescription>{t('model.dialog.rerankWindow.tokenizerHelp')}</FormDescription>
                            <FormMessage />
                        </FormItem>
                    )} />
                </div>
            )}
        </div>
    )
}
