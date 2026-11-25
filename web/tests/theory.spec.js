import { expect, test } from '@playwright/test'

for (const width of [1440, 390]) {
  test(`explore Nav2, the nested training loop and export at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 })
    await page.goto('/docs/architecture/')
    const diagram = page.getByRole('region', { name: 'Interactive system architecture' })
    const controller = diagram.getByRole('button', { name: /^DWARL / })
    await controller.focus()
    await page.keyboard.press('Enter')
    await expect(diagram.getByRole('heading', { name: 'Inputs, actions and training loop' })).toBeVisible()
    const learner = diagram.getByRole('button', { name: /^PPO/ })
    await learner.click()
    const dialog = page.getByRole('dialog')
    await expect(dialog.getByRole('heading', { name: 'Critic & PPO' })).toBeVisible()
    await dialog.getByRole('button', { name: /Checkpoint selection/ }).click()
    await dialog.getByRole('button', { name: /ONNX bundle/ }).click()
    await expect(dialog.getByRole('heading', { name: 'ONNX bundle' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(learner).toBeFocused()
    await diagram.getByRole('button', { name: /← Nav2 integration/ }).click()
    await expect(controller).toBeFocused()
    await diagram.getByRole('button', { name: /Collision monitor/ }).click()
    await expect(dialog.locator('code').first()).toHaveText('cmd_vel_raw')
  })
}
