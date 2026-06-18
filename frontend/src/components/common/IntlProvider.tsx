/**
 * Client-side Internationalization Provider
 */

'use client';

import React from 'react';
import { NextIntlClientProvider } from 'next-intl';
import { defaultLocale } from '@/i18n/config';

import enMessages from '../../../messages/en.json';

interface IntlProviderProps {
  children: React.ReactNode;
}

export function IntlProvider({ children }: IntlProviderProps) {
  return (
    <NextIntlClientProvider locale={defaultLocale} messages={enMessages} timeZone="UTC">
      {children}
    </NextIntlClientProvider>
  );
}
